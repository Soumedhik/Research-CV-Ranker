"""
analyzers/paper_analyzer.py
Enriches publications by querying Semantic Scholar & OpenAlex.
Infers venue ranks (A*, A, B, C) and citation counts.
"""

import re
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import Config

console = Console()

# Singleton config for venue matching
_config = Config()

RANK_ORDER = {"A*": 4, "A": 3, "B": 2, "C": 1, "?": 0}


@dataclass
class EnrichedPaper:
    title: str
    venue: str
    year: Optional[int]
    venue_rank: str          # A*, A, B, C, ?
    venue_rank_source: str   # "known_list", "inferred", "unknown"
    citation_count: int
    is_first_author: bool
    semantic_scholar_id: Optional[str]
    semantic_scholar_url: Optional[str]
    influential_citation_count: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    open_access: bool = False


class VenueRanker:
    """Infers venue rank from name matching against known lists."""

    def __init__(self):
        self.cfg = _config

    def rank(self, venue: str) -> tuple[str, str]:
        """Returns (rank, source)."""
        if not venue:
            return "?", "unknown"
        venue_lower = venue.lower()
        venue_lower = re.sub(r"\d{4}", "", venue_lower).strip()  # remove years

        for v in self.cfg.astar_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "A*", "known_list"

        for v in self.cfg.a_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "A", "known_list"

        for v in self.cfg.b_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "B", "known_list"

        # Heuristic inferences
        venue_up = venue.upper()
        if any(x in venue_up for x in ["NATURE", "SCIENCE", "CELL", "NEJM", "LANCET"]):
            return "A*", "inferred"
        if any(x in venue_up for x in ["IEEE TRANS", "ACM TRANS", "SPRINGER"]):
            return "A", "inferred"
        if any(x in venue_up for x in ["WORKSHOP", "WS ", "ARXIV"]):
            return "C", "inferred"
        if "PREPRINT" in venue_up:
            return "C", "inferred"

        return "?", "unknown"


class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,year,venue,citationCount,influentialCitationCount,authors,isOpenAccess,s2FieldsOfStudy,externalIds"

    def __init__(self, delay: float = 1.2):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ResearchRank/1.0 (academic tool)"})
        self.delay = delay

    def search(self, title: str) -> Optional[dict]:
        try:
            time.sleep(self.delay)
            resp = self.session.get(
                f"{self.BASE}/paper/search",
                params={"query": title, "fields": self.FIELDS, "limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    return data["data"][0]
        except Exception:
            pass
        return None


class PaperAnalyzer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ranker = VenueRanker()
        self.ss_client = SemanticScholarClient()

    def enrich_all(self, resumes: list) -> list:
        """Add enriched_papers to each AnalyzedResume."""
        total_papers = sum(len(r.publications) for r in resumes if not r.llm_error)
        if total_papers == 0:
            console.print("[yellow]⚠ No publications found to look up[/yellow]\n")
            return resumes

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Looking up papers[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=total_papers)
            for resume in resumes:
                if resume.llm_error:
                    continue
                enriched = []
                for pub in resume.publications:
                    progress.update(task, description=f"[dim]{pub.title[:50]}[/dim]")
                    ep = self._enrich_paper(pub)
                    enriched.append(ep.__dict__)
                    progress.advance(task)
                resume.enriched_papers = enriched

        console.print(f"[green]✓[/green] Looked up [bold]{total_papers}[/bold] papers\n")
        return resumes

    def _enrich_paper(self, pub) -> EnrichedPaper:
        rank, rank_source = self.ranker.rank(pub.venue)
        ep = EnrichedPaper(
            title=pub.title,
            venue=pub.venue,
            year=pub.year,
            venue_rank=rank,
            venue_rank_source=rank_source,
            citation_count=0,
            is_first_author=pub.is_first_author,
            semantic_scholar_id=None,
            semantic_scholar_url=None,
        )

        # Try Semantic Scholar lookup
        ss_data = self.ss_client.search(pub.title)
        if ss_data:
            ep.citation_count = ss_data.get("citationCount", 0) or 0
            ep.influential_citation_count = ss_data.get("influentialCitationCount", 0) or 0
            ep.open_access = ss_data.get("isOpenAccess", False)
            ep.semantic_scholar_id = ss_data.get("paperId")
            if ep.semantic_scholar_id:
                ep.semantic_scholar_url = f"https://www.semanticscholar.org/paper/{ep.semantic_scholar_id}"
            fos = ss_data.get("s2FieldsOfStudy", [])
            ep.fields_of_study = list({f["category"] for f in fos if f.get("category")})

            # Use SS venue if we didn't recognize ours
            if rank == "?" and ss_data.get("venue"):
                ss_venue = ss_data["venue"]
                ss_rank, ss_source = self.ranker.rank(ss_venue)
                if ss_rank != "?":
                    ep.venue = ss_venue
                    ep.venue_rank = ss_rank
                    ep.venue_rank_source = ss_source

        return ep
