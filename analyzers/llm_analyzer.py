"""
analyzers/llm_analyzer.py
Uses Groq LLM to extract structured information from raw resume text.
Includes proactive rate limiting, token tracking, and exponential backoff.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from groq import Groq
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import Config
from parsers.resume_parser import RawResume
from utils.rate_limiter import GroqRateLimiter, estimate_tokens, backoff_delay

console = Console()


@dataclass
class ResearchExperience:
    role: str
    organization: str
    duration: str
    start_year: Optional[int]
    end_year: Optional[int]  # None = present
    description: str
    is_academic: bool


@dataclass
class Education:
    degree: str           # PhD, MS, BS, etc.
    field: str
    institution: str
    year: Optional[int]
    gpa: Optional[str]


@dataclass
class Publication:
    title: str
    venue: str
    year: Optional[int]
    authors: list[str]
    is_first_author: bool
    raw_citation: str


@dataclass
class AnalyzedResume:
    # Source
    raw: RawResume

    # Extracted fields
    name: str = ""
    email: str = ""
    homepage: Optional[str] = None

    education: list[Education] = field(default_factory=list)
    research_experience: list[ResearchExperience] = field(default_factory=list)
    publications: list[Publication] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    awards: list[str] = field(default_factory=list)

    # LLM-generated summaries
    research_summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    fit_notes: str = ""      # Only populated when job_description given

    # Enriched later by PaperAnalyzer
    enriched_papers: list[dict] = field(default_factory=list)

    # Academic metrics (computed by Scorer after enrichment)
    h_index: int = 0
    total_citations: int = 0
    avg_citations: float = 0.0
    years_active: int = 0            # span of publishing years
    venue_tier_counts: dict = field(default_factory=dict)   # {"A*":2,"A":3,...}

    # Scores (set by Scorer)
    scores: dict = field(default_factory=dict)
    total_score: float = 0.0
    rank: int = 0

    llm_error: Optional[str] = None


# ── Optimized prompt (concise to reduce token usage) ──────────────────────────

EXTRACTION_PROMPT = """Extract structured data from this CV. Return ONLY valid JSON, no markdown fences.

Schema:
{{"name":"str","email":"str","homepage":"str or null",
"education":[{{"degree":"PhD/MS/BS/etc","field":"str","institution":"str","year":int,"gpa":"str or null"}}],
"research_experience":[{{"role":"str","organization":"str","duration":"str","start_year":int,"end_year":int or null,"description":"brief str","is_academic":bool}}],
"publications":[{{"title":"str","venue":"str","year":int,"authors":["str"],"is_first_author":bool,"raw_citation":"str"}}],
"skills":["str"],"awards":["str"],
"research_summary":"2-3 sentences on research focus",
"strengths":["str","str"],"weaknesses":["str","str"]{job_schema}}}
{job_section}
CV TEXT:
{cv_text}"""


class LLMAnalyzer:
    def __init__(self, config: Config, verbose: bool = False,
                 rate_limiter: Optional[GroqRateLimiter] = None):
        self.config = config
        self.verbose = verbose
        self.rate_limiter = rate_limiter
        self._init_client()

    def _init_client(self):
        try:
            config = self.config
            config.validate()
            self.client = Groq(api_key=config.groq_api_key)
        except ValueError as e:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {e}\n")
            sys.exit(1)

    def analyze_all(self, resumes: list[RawResume],
                    job_description: Optional[str] = None) -> list["AnalyzedResume"]:
        results = []
        valid_count = sum(1 for r in resumes if r.is_valid)
        model_name = self.config.groq_model.split("/")[-1]
        limits = self.rate_limiter.limits if self.rate_limiter else {}
        tpm_str = f"{limits.get('tpm', '?'):,}" if limits else "?"
        tpd_str = f"{limits.get('tpd', '?'):,}" if limits else "?"

        console.print(
            f"[dim]  Model: [cyan]{self.config.groq_model}[/cyan]  |  "
            f"TPM: {tpm_str}  |  TPD: {tpd_str}  |  "
            f"Calls needed: {valid_count}[/dim]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing with Groq LLM[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(resumes))
            for resume in resumes:
                progress.update(task, description=f"[dim]{resume.filename}[/dim]")
                if not resume.is_valid:
                    analyzed = AnalyzedResume(raw=resume, llm_error=resume.error)
                    analyzed.name = resume.filename
                    results.append(analyzed)
                else:
                    analyzed = self._analyze_one(resume, job_description)
                    results.append(analyzed)
                progress.advance(task)

        valid = sum(1 for r in results if not r.llm_error)
        console.print(f"[green]✓[/green] LLM analyzed [bold]{valid}/{len(resumes)}[/bold] resumes\n")

        # Show token usage summary
        if self.rate_limiter:
            s = self.rate_limiter.stats
            console.print(
                f"[dim]  Tokens: {s.total_tokens:,} used "
                f"({s.total_prompt_tokens:,} prompt + {s.total_completion_tokens:,} completion)  |  "
                f"Avg/call: {s.avg_tokens_per_request:,.0f}  |  "
                f"Waits: {s.rate_limit_waits}  |  Retries: {s.retries}[/dim]\n"
            )

        return results

    def _analyze_one(self, resume: RawResume,
                     job_description: Optional[str]) -> "AnalyzedResume":
        # Truncate CV text to save tokens (configurable)
        cv_text = resume.text[:self.config.groq_cv_truncate_chars]

        job_section = ""
        job_schema = ""
        if job_description:
            job_section = f'Assess fit for this role: {job_description}'
            job_schema = ',"fit_notes":"2-3 sentence fit assessment"'

        prompt = EXTRACTION_PROMPT.format(
            cv_text=cv_text,
            job_section=job_section,
            job_schema=job_schema,
        )

        try:
            response = self._call_groq(prompt)
            data = self._parse_json(response)
            analyzed = self._build_analyzed(resume, data)
            # Second pass: regex-based extraction to catch missing papers
            extra_pubs = self._regex_extract_publications(cv_text, analyzed.publications)
            if extra_pubs:
                analyzed.publications.extend(extra_pubs)
                if self.verbose:
                    console.print(
                        f"  [dim]  +{len(extra_pubs)} paper(s) recovered by second-pass regex[/dim]"
                    )
            return analyzed
        except Exception as e:
            if self.verbose:
                console.print(f"  [red]LLM error for {resume.filename}: {e}[/red]")
            return AnalyzedResume(
                raw=resume,
                name=resume.filename,
                llm_error=str(e)
            )

    def _call_groq(self, prompt: str, model: Optional[str] = None,
                   attempt: int = 0) -> str:
        """
        Call Groq API with proactive rate limiting and exponential backoff.
        Falls back to secondary model on repeated rate-limit errors.
        """
        model = model or self.config.groq_model
        max_retries = self.config.groq_max_retries

        # Proactive rate limiting — wait before sending if needed
        if self.rate_limiter:
            estimated_input = estimate_tokens(prompt)
            self.rate_limiter.wait_if_needed(estimated_input)

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.groq_max_tokens,
                temperature=self.config.groq_temperature,
            )

            # Record actual token usage from response
            if self.rate_limiter and resp.usage:
                self.rate_limiter.record_usage(
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens,
                )

            return resp.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
            is_transient = is_rate_limit or "503" in error_str or "timeout" in error_str.lower()

            if is_transient and attempt < max_retries:
                delay = backoff_delay(attempt)
                if self.rate_limiter:
                    self.rate_limiter.stats.retries += 1

                if self.verbose:
                    console.print(
                        f"  [yellow]⚠ {'Rate limited' if is_rate_limit else 'Transient error'}, "
                        f"retry {attempt + 1}/{max_retries} in {delay:.1f}s[/yellow]"
                    )
                time.sleep(delay)

                # Switch to fallback model after first retry
                next_model = model
                if is_rate_limit and attempt >= 1:
                    next_model = self.config.groq_fallback_model
                    if self.rate_limiter:
                        self.rate_limiter.update_model(next_model)
                    if self.verbose:
                        console.print(f"  [yellow]  → Falling back to {next_model}[/yellow]")

                return self._call_groq(prompt, model=next_model, attempt=attempt + 1)
            raise

    def _parse_json(self, text: str) -> dict:
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Attempt 1: strict parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract the FIRST complete JSON object
        # (handles "Extra data" where LLM appended text after the JSON)
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    break

        # Attempt 3: find ANY JSON object via regex (last resort)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")

    def _build_analyzed(self, resume: RawResume, data: dict) -> "AnalyzedResume":
        analyzed = AnalyzedResume(raw=resume)
        analyzed.name = data.get("name", resume.filename)
        analyzed.email = data.get("email", "")
        analyzed.homepage = data.get("homepage")

        for e in data.get("education", []):
            analyzed.education.append(Education(
                degree=e.get("degree", ""),
                field=e.get("field", ""),
                institution=e.get("institution", ""),
                year=e.get("year"),
                gpa=e.get("gpa"),
            ))

        for r in data.get("research_experience", []):
            analyzed.research_experience.append(ResearchExperience(
                role=r.get("role", ""),
                organization=r.get("organization", ""),
                duration=r.get("duration", ""),
                start_year=r.get("start_year"),
                end_year=r.get("end_year"),
                description=r.get("description", ""),
                is_academic=r.get("is_academic", True),
            ))

        for p in data.get("publications", []):
            analyzed.publications.append(Publication(
                title=p.get("title", ""),
                venue=p.get("venue", ""),
                year=p.get("year"),
                authors=p.get("authors", []),
                is_first_author=p.get("is_first_author", False),
                raw_citation=p.get("raw_citation", ""),
            ))

        analyzed.skills = data.get("skills", [])
        analyzed.awards = data.get("awards", [])
        analyzed.research_summary = data.get("research_summary", "")
        analyzed.strengths = data.get("strengths", [])
        analyzed.weaknesses = data.get("weaknesses", [])
        analyzed.fit_notes = data.get("fit_notes", "")

        return analyzed

    # ── Second-Pass Publication Extraction ─────────────────────────────────────

    # Common citation patterns:
    #   [1] Author et al., "Title", Venue, Year.
    #   Author, A., Author, B. (Year). Title. Venue, ...
    #   Author et al. Title. In Proc. of Venue, Year.
    _CITATION_PATTERNS = [
        # Numbered: [1] ... "Title" ... Venue ... 20XX
        re.compile(
            r'\[\d+\]\s*'
            r'(?P<authors>[^"]+?)'
            r'["\u201c](?P<title>[^"\u201d]{15,})["\u201d]'
            r'[,.]\s*(?P<rest>.+?)'
            r'(?P<year>(?:19|20)\d{2})',
            re.IGNORECASE,
        ),
        # APA-ish: Author (Year). Title. Venue.
        re.compile(
            r'(?P<authors>[A-Z][a-z]+(?:,?\s+(?:and\s+)?[A-Z]\.?[a-z]*)+)'
            r'\s*\((?P<year>(?:19|20)\d{2})\)\.?\s*'
            r'(?P<title>[^.]{15,}?)\.'  
            r'\s*(?P<rest>.+)',
            re.IGNORECASE,
        ),
        # Generic: Title. In Proceedings of / In Venue / Journal Name, Year
        re.compile(
            r'(?P<title>[A-Z][^.]{15,}?)\.\s*'
            r'(?:In\s+)?(?:Proceedings\s+of\s+(?:the\s+)?)?'
            r'(?P<rest>[A-Z][^.]+?)'
            r'[,.]\s*(?P<year>(?:19|20)\d{2})',
            re.IGNORECASE,
        ),
    ]

    def _regex_extract_publications(
        self,
        cv_text: str,
        existing_pubs: list[Publication],
    ) -> list[Publication]:
        """
        Regex second-pass to catch publications the LLM missed.
        Finds citation-like patterns in the raw text, deduplicates against
        the LLM-extracted list using title similarity.
        """
        existing_titles = [p.title.lower().strip() for p in existing_pubs]
        new_pubs: list[Publication] = []

        for pattern in self._CITATION_PATTERNS:
            for m in pattern.finditer(cv_text):
                title = m.group("title").strip()
                # Skip very short or very long "titles" (likely noise)
                if len(title) < 15 or len(title) > 300:
                    continue

                # Deduplicate: skip if similar to any existing publication
                if self._is_duplicate(title, existing_titles):
                    continue

                year_str = m.group("year") if "year" in m.groupdict() else None
                year = int(year_str) if year_str else None

                rest = m.group("rest") if "rest" in m.groupdict() else ""
                venue = self._extract_venue_from_rest(rest)

                authors_str = m.group("authors") if "authors" in m.groupdict() else ""
                authors = [a.strip() for a in re.split(r",\s*|\s+and\s+", authors_str) if a.strip()]

                pub = Publication(
                    title=title,
                    venue=venue,
                    year=year,
                    authors=authors[:10],  # cap
                    is_first_author=False,  # can't reliably determine from regex
                    raw_citation=m.group(0)[:300],
                )
                new_pubs.append(pub)
                existing_titles.append(title.lower().strip())  # prevent self-duplication

        return new_pubs

    def _is_duplicate(self, title: str, existing_titles: list[str]) -> bool:
        """Check if title is similar enough to an existing one (>0.7 ratio)."""
        title_lower = title.lower().strip()
        for et in existing_titles:
            ratio = SequenceMatcher(None, title_lower, et).ratio()
            if ratio > 0.7:
                return True
        return False

    @staticmethod
    def _extract_venue_from_rest(rest: str) -> str:
        """Try to extract a venue name from the rest of a citation string."""
        if not rest:
            return ""
        # Common venue prefixes
        venue_match = re.match(
            r'(?:In\s+)?(?:Proceedings\s+of\s+(?:the\s+)?)?'
            r'([A-Z][^,.:;]{3,60})',
            rest.strip(),
            re.IGNORECASE,
        )
        if venue_match:
            return venue_match.group(1).strip()
        return rest.strip()[:60]
