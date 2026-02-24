"""
analyzers/paper_analyzer.py
Enriches publications by querying Semantic Scholar, OpenAlex, and Crossref.
Infers venue ranks (A*, A, B, C) using bundled CORE/Scimago datasets,
config-based venue lists, and heuristics. Caches results to avoid
redundant API calls across runs.
"""

import re
import time
import urllib.parse
import requests
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import Config
from utils.venue_data import get_venue_dataset
from utils.cache import PaperCache, VenueCache

console = Console()

# Singleton config for venue matching
_config = Config()

RANK_ORDER = {"A*": 4, "A": 3, "B": 2, "C": 1, "?": 0}


@dataclass
class EnrichedPaper:
    title: str
    venue: str
    venue_canonical: str     # Normalized venue name
    year: Optional[int]
    venue_rank: str          # A*, A, B, C, ?
    venue_rank_source: str   # "dataset", "config_list", "heuristic", "unknown"
    citation_count: int
    is_first_author: bool
    semantic_scholar_id: Optional[str]
    semantic_scholar_url: Optional[str]
    influential_citation_count: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    open_access: bool = False
    lookup_source: str = ""  # "semantic_scholar", "openalex", "crossref", "none"


class VenueRanker:
    """
    Multi-layer venue ranking:
      1. Bundled datasets (CORE conferences, Scimago journals) — ~800+ venues
      2. Config-based curated lists (astar_venues, a_venues, b_venues)
      3. Heuristic rules (keyword-based fallback)
    Results are cached to disk for speed across runs.
    """

    def __init__(self, use_cache: bool = True):
        self.cfg = _config
        self.dataset = get_venue_dataset()
        self._venue_cache = VenueCache() if use_cache else None
        if self._venue_cache:
            console.print(
                f"[dim]  Venue dataset: {self.dataset.total_venues} venues, "
                f"{self.dataset.total_aliases} aliases loaded[/dim]"
            )

    def rank(self, venue: str) -> tuple[str, str, str]:
        """
        Returns (rank, source, canonical_name).
        Tries each layer in order until a rank is found.
        """
        if not venue:
            return "?", "unknown", venue or ""

        # Check cache first
        if self._venue_cache:
            cached = self._venue_cache.get(venue)
            if cached:
                return cached["rank"], cached["source"], cached["canonical"]

        # Normalize venue name using alias dictionary
        canonical = self.dataset.normalize(venue)

        # Layer 1: Bundled dataset (CORE + Scimago)
        rank, source = self.dataset.lookup(venue)
        if rank != "?":
            self._cache_venue(venue, rank, f"dataset:{source}", canonical)
            return rank, f"dataset:{source}", canonical

        # Also try the canonical name directly
        if canonical != venue:
            rank, source = self.dataset.lookup(canonical)
            if rank != "?":
                self._cache_venue(venue, rank, f"dataset:{source}", canonical)
                return rank, f"dataset:{source}", canonical

        # Layer 2: Config-based curated lists (backward compatible)
        rank, source = self._rank_from_config(venue)
        if rank != "?":
            self._cache_venue(venue, rank, source, canonical)
            return rank, source, canonical

        # Layer 3: Heuristic rules
        rank, source = self._rank_heuristic(venue)
        self._cache_venue(venue, rank, source, canonical)
        return rank, source, canonical

    def _rank_from_config(self, venue: str) -> tuple[str, str]:
        """Match against config.py venue lists."""
        venue_lower = venue.lower()
        venue_lower = re.sub(r"\d{4}", "", venue_lower).strip()

        for v in self.cfg.astar_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "A*", "config_list"
        for v in self.cfg.a_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "A", "config_list"
        for v in self.cfg.b_venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return "B", "config_list"
        return "?", "unknown"

    def _rank_heuristic(self, venue: str) -> tuple[str, str]:
        """Keyword-based heuristic fallback for unrecognized venues."""
        venue_up = venue.upper()

        # A* signals
        if any(x in venue_up for x in [
            "NATURE", "SCIENCE", "CELL", "NEJM", "LANCET", "PNAS",
            "JAMA", "BMJ",
        ]):
            return "A*", "heuristic"

        # A signals
        if any(x in venue_up for x in [
            "IEEE TRANS", "ACM TRANS", "IEEE JOURNAL",
            "TRANSACTIONS ON",
        ]):
            return "A", "heuristic"

        # B signals
        if any(x in venue_up for x in [
            "IEEE", "ACM", "SPRINGER", "ELSEVIER",
            "INTERNATIONAL CONFERENCE", "INTERNATIONAL JOURNAL",
        ]):
            return "B", "heuristic"

        # C signals
        if any(x in venue_up for x in [
            "WORKSHOP", "WS ", "ARXIV", "PREPRINT", "BIORXIV", "MEDRXIV",
            "TECHREPORT", "TECHNICAL REPORT", "THESIS",
            "POSTER", "DEMO", "EXTENDED ABSTRACT",
        ]):
            return "C", "heuristic"

        return "?", "unknown"

    def _cache_venue(self, venue: str, rank: str, source: str, canonical: str):
        if self._venue_cache:
            self._venue_cache.put(venue, {
                "rank": rank, "source": source, "canonical": canonical
            })

    def flush_cache(self):
        if self._venue_cache:
            self._venue_cache.flush()


# ── API Clients ───────────────────────────────────────────────────────────────

class SemanticScholarClient:
    """Semantic Scholar Academic Graph API (free, no key)."""
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = (
        "title,year,venue,citationCount,influentialCitationCount,"
        "authors,isOpenAccess,s2FieldsOfStudy,externalIds"
    )

    def __init__(self, delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchRank/2.0 (academic-cv-ranker; github.com/Soumedhik/Research-CV-Ranker)"
        })
        self.delay = delay
        self.requests_made = 0
        self._consecutive_429 = 0

    def search(self, title: str, _retry: int = 0) -> Optional[dict]:
        try:
            time.sleep(self.delay)
            self.requests_made += 1
            resp = self.session.get(
                f"{self.BASE}/paper/search",
                params={"query": title, "fields": self.FIELDS, "limit": 3},
                timeout=8,
            )
            if resp.status_code == 200:
                self._consecutive_429 = 0
                data = resp.json()
                results = data.get("data", [])
                if results:
                    return self._best_match(title, results)
            elif resp.status_code == 429:
                self._consecutive_429 += 1
                if _retry < 2:
                    time.sleep(5 * (self._consecutive_429))
                    return self.search(title, _retry=_retry + 1)
                return None
        except Exception:
            pass
        return None

    def _best_match(self, query_title: str, results: list[dict]) -> Optional[dict]:
        """Pick the result that best matches the query title."""
        query_lower = query_title.strip().lower()
        best = None
        best_score = -1
        for r in results:
            r_title = (r.get("title") or "").strip().lower()
            # Simple overlap ratio
            score = self._title_similarity(query_lower, r_title)
            if score > best_score:
                best_score = score
                best = r
        # Require at least 50% similarity to accept
        return best if best_score > 0.5 else (results[0] if results else None)

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Word-overlap similarity between two titles."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a), len(words_b))


class OpenAlexClient:
    """OpenAlex API fallback (free, no key, polite pool with email)."""
    BASE = "https://api.openalex.org"

    def __init__(self, email: str = "researchrank@lab.local", delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"ResearchRank/2.0 (mailto:{email})"
        })
        self.email = email
        self.delay = delay
        self.requests_made = 0

    def search(self, title: str) -> Optional[dict]:
        """Search OpenAlex for a paper by title."""
        try:
            time.sleep(self.delay)
            self.requests_made += 1
            # Use the works search endpoint
            resp = self.session.get(
                f"{self.BASE}/works",
                params={
                    "search": title,
                    "per_page": 1,
                    "mailto": self.email,
                },
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return self._normalize(results[0])
        except Exception:
            pass
        return None

    def _normalize(self, work: dict) -> dict:
        """Convert OpenAlex work format to our standard format."""
        # Extract venue from primary location
        venue = ""
        primary_loc = work.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        venue = source.get("display_name", "")

        # Citation count
        cites = work.get("cited_by_count", 0) or 0

        # Open access
        oa = work.get("open_access", {})
        is_oa = oa.get("is_oa", False)

        # Fields of study (concepts in OpenAlex)
        concepts = work.get("concepts", [])
        fields = [c.get("display_name", "") for c in concepts[:5] if c.get("level", 99) <= 1]

        # Authors
        authorships = work.get("authorships", [])
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in authorships
        ]

        return {
            "title": work.get("display_name", work.get("title", "")),
            "year": work.get("publication_year"),
            "venue": venue,
            "citationCount": cites,
            "influentialCitationCount": 0,  # OpenAlex doesn't have this
            "isOpenAccess": is_oa,
            "authors": [{"name": a} for a in authors],
            "s2FieldsOfStudy": [{"category": f} for f in fields],
            "paperId": None,
            "externalIds": work.get("ids", {}),
            "source": "openalex",
            "openalex_id": work.get("id", ""),
        }


class CrossrefClient:
    """Crossref API fallback (free, no key)."""
    BASE = "https://api.crossref.org"

    def __init__(self, email: str = "researchrank@lab.local", delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"ResearchRank/2.0 (mailto:{email})"
        })
        self.email = email
        self.delay = delay
        self.requests_made = 0

    def search(self, title: str) -> Optional[dict]:
        """Search Crossref for a paper by title."""
        try:
            time.sleep(self.delay)
            self.requests_made += 1
            resp = self.session.get(
                f"{self.BASE}/works",
                params={
                    "query.bibliographic": title,
                    "rows": 1,
                    "mailto": self.email,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("message", {}).get("items", [])
                if items:
                    return self._normalize(items[0])
        except Exception:
            pass
        return None

    def _normalize(self, item: dict) -> dict:
        """Convert Crossref item to our standard format."""
        # Title
        titles = item.get("title", [])
        title_str = titles[0] if titles else ""

        # Venue
        containers = item.get("container-title", [])
        venue = containers[0] if containers else ""

        # Year
        issued = item.get("issued", {})
        date_parts = issued.get("date-parts", [[None]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        # Citation count
        cites = item.get("is-referenced-by-count", 0) or 0

        # Authors
        authors_raw = item.get("author", [])
        authors = [
            {"name": f"{a.get('given', '')} {a.get('family', '')}".strip()}
            for a in authors_raw
        ]

        return {
            "title": title_str,
            "year": year,
            "venue": venue,
            "citationCount": cites,
            "influentialCitationCount": 0,
            "isOpenAccess": item.get("is-referenced-by-count") is not None,  # rough proxy
            "authors": authors,
            "s2FieldsOfStudy": [],
            "paperId": None,
            "externalIds": {"DOI": item.get("DOI", "")},
            "source": "crossref",
        }


# ── Main Paper Analyzer ──────────────────────────────────────────────────────

class PaperAnalyzer:
    """
    Enriches candidate publications with external data and venue rankings.

    Pipeline per paper:
      1. Check paper cache (skip API if cached)
      2. Query Semantic Scholar
      3. If not found → try OpenAlex
      4. If not found → try Crossref
      5. Rank venue using dataset + config + heuristics
      6. Cache result
    """

    def __init__(self, config: Optional[Config] = None, verbose: bool = False):
        self.config = config or Config()
        self.verbose = verbose
        self.ranker = VenueRanker(use_cache=True)
        self.ss_client = SemanticScholarClient(delay=self.config.semantic_scholar_delay)
        self.oa_client = OpenAlexClient(email=self.config.openalex_email)
        self.cr_client = CrossrefClient(email=self.config.openalex_email)
        self._paper_cache = PaperCache()
        self._enrichment_stats = {
            "total": 0, "cached": 0, "semantic_scholar": 0,
            "openalex": 0, "crossref": 0, "not_found": 0,
        }

    def enrich_all(self, resumes: list) -> list:
        """Add enriched_papers to each AnalyzedResume."""
        total_papers = sum(len(r.publications) for r in resumes if not r.llm_error)
        if total_papers == 0:
            console.print("[yellow]⚠ No publications found to look up[/yellow]\n")
            return resumes

        self._enrichment_stats["total"] = total_papers

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Enriching papers[/bold blue]"),
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
                    short_title = pub.title[:50]
                    progress.update(task, description=f"[dim]{short_title}[/dim]")
                    ep = self._enrich_paper(pub)
                    enriched.append(ep.__dict__)
                    progress.advance(task)
                resume.enriched_papers = enriched

        # Flush caches to disk
        self._paper_cache.flush()
        self.ranker.flush_cache()

        # Print enrichment summary
        s = self._enrichment_stats
        found = s["semantic_scholar"] + s["openalex"] + s["crossref"] + s["cached"]
        coverage = (found / max(s["total"], 1)) * 100
        console.print(
            f"[green]✓[/green] Enriched [bold]{total_papers}[/bold] papers "
            f"([cyan]{coverage:.0f}%[/cyan] coverage)\n"
            f"[dim]  Sources: SS={s['semantic_scholar']} | "
            f"OpenAlex={s['openalex']} | Crossref={s['crossref']} | "
            f"Cached={s['cached']} | Not found={s['not_found']}[/dim]\n"
        )

        return resumes

    def _enrich_paper(self, pub) -> EnrichedPaper:
        """Enrich a single publication with external data."""
        # Check paper cache first
        cached = self._paper_cache.get(pub.title)
        if cached:
            self._enrichment_stats["cached"] += 1
            # Re-rank venue (dataset may have been updated since cache)
            rank, rank_src, canonical = self.ranker.rank(cached.get("venue", pub.venue))
            return EnrichedPaper(
                title=cached.get("title", pub.title),
                venue=cached.get("venue", pub.venue),
                venue_canonical=canonical,
                year=cached.get("year", pub.year),
                venue_rank=rank,
                venue_rank_source=rank_src,
                citation_count=cached.get("citation_count", 0),
                is_first_author=pub.is_first_author,
                semantic_scholar_id=cached.get("semantic_scholar_id"),
                semantic_scholar_url=cached.get("semantic_scholar_url"),
                influential_citation_count=cached.get("influential_citation_count", 0),
                fields_of_study=cached.get("fields_of_study", []),
                open_access=cached.get("open_access", False),
                lookup_source=cached.get("lookup_source", "cached"),
            )

        # Initial venue ranking (may be improved after API lookup)
        rank, rank_source, canonical = self.ranker.rank(pub.venue)

        ep = EnrichedPaper(
            title=pub.title,
            venue=pub.venue,
            venue_canonical=canonical,
            year=pub.year,
            venue_rank=rank,
            venue_rank_source=rank_source,
            citation_count=0,
            is_first_author=pub.is_first_author,
            semantic_scholar_id=None,
            semantic_scholar_url=None,
            lookup_source="none",
        )

        # Try Semantic Scholar → OpenAlex → Crossref
        api_data = self._lookup_paper(pub.title)

        if api_data:
            self._apply_api_data(ep, api_data, rank, rank_source)
        else:
            self._enrichment_stats["not_found"] += 1

        # Cache the enriched result
        self._paper_cache.put(pub.title, {
            "title": ep.title,
            "venue": ep.venue,
            "year": ep.year,
            "citation_count": ep.citation_count,
            "influential_citation_count": ep.influential_citation_count,
            "open_access": ep.open_access,
            "fields_of_study": ep.fields_of_study,
            "semantic_scholar_id": ep.semantic_scholar_id,
            "semantic_scholar_url": ep.semantic_scholar_url,
            "lookup_source": ep.lookup_source,
        })

        return ep

    def _lookup_paper(self, title: str) -> Optional[dict]:
        """
        Try multiple APIs in order:
          1. Semantic Scholar (best data quality)
          2. OpenAlex (broadest coverage)
          3. Crossref (fallback, DOI-focused)
        """
        # --- Semantic Scholar ---
        ss_data = self.ss_client.search(title)
        if ss_data and ss_data.get("title"):
            ss_data["source"] = "semantic_scholar"
            self._enrichment_stats["semantic_scholar"] += 1
            return ss_data

        # --- OpenAlex ---
        try:
            oa_data = self.oa_client.search(title)
            if oa_data and oa_data.get("title"):
                self._enrichment_stats["openalex"] += 1
                return oa_data
        except Exception:
            pass

        # --- Crossref ---
        try:
            cr_data = self.cr_client.search(title)
            if cr_data and cr_data.get("title"):
                self._enrichment_stats["crossref"] += 1
                return cr_data
        except Exception:
            pass

        return None

    def _apply_api_data(self, ep: EnrichedPaper, data: dict,
                        original_rank: str, original_source: str):
        """Apply API lookup data to an EnrichedPaper."""
        ep.citation_count = data.get("citationCount", 0) or 0
        ep.influential_citation_count = data.get("influentialCitationCount", 0) or 0
        ep.open_access = data.get("isOpenAccess", False)
        ep.lookup_source = data.get("source", "unknown")

        # Semantic Scholar specific
        if data.get("paperId"):
            ep.semantic_scholar_id = data["paperId"]
            ep.semantic_scholar_url = f"https://www.semanticscholar.org/paper/{data['paperId']}"

        # OpenAlex specific
        if data.get("openalex_id"):
            ep.semantic_scholar_url = ep.semantic_scholar_url or data["openalex_id"]

        # Fields of study
        fos = data.get("s2FieldsOfStudy", [])
        ep.fields_of_study = list({f["category"] for f in fos if f.get("category")})

        # Year from API if missing
        if not ep.year and data.get("year"):
            ep.year = data["year"]

        # Try to improve venue ranking using API venue data
        if original_rank == "?" and data.get("venue"):
            api_venue = data["venue"]
            api_rank, api_src, api_canonical = self.ranker.rank(api_venue)
            if api_rank != "?":
                ep.venue = api_venue
                ep.venue_canonical = api_canonical
                ep.venue_rank = api_rank
                ep.venue_rank_source = api_src

    def get_enrichment_stats(self) -> dict:
        """Return paper enrichment statistics."""
        return dict(self._enrichment_stats)
