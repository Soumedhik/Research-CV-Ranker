# 🔬 ResearchRank

> Mass-ranking academic CVs/resumes for research labs — terminal-native, LLM-powered, fully free.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20%28free%29-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![APIs](https://img.shields.io/badge/APIs-Semantic%20Scholar%20%7C%20OpenAlex%20%7C%20Crossref-green)

---

## What it does

Feed it N resumes (PDF, DOCX, TXT, Markdown). Get back a ranked leaderboard with:

- **Structured extraction** — name, education, research roles, publications, skills, awards via Groq LLM
- **Second-pass publication recovery** — regex patterns catch papers the LLM missed; deduplication by title similarity
- **Publication enrichment** — looks up each paper on **Semantic Scholar** → **OpenAlex** → **Crossref** (cascading fallback; cached to disk)
- **Dataset-backed venue ranking** — 800+ venues from bundled **CORE Conference Rankings** + **Scimago Journal Rank** (Q1→A\*, Q2→A, Q3→B, Q4→C) + config lists + heuristics
- **Venue name normalization** — 300+ aliases map variant names to canonical forms (e.g., "Conference on Neural Information Processing Systems" → "NeurIPS")
- **Publication timeline** — ASCII bar chart showing research activity per year with venue-rank annotations
- **Multi-dimensional scoring** — Publications (30%), Research (25%), Education (20%), Role Fit (15%), Trajectory (10%) — all configurable
- **Proactive rate limiting** — RPM/TPM/RPD/TPD tracking with sliding windows; never hits 429
- **Rich terminal output** — leaderboard table, per-candidate deep-dive cards, SWOT analysis, score bars
- **JSON export** — machine-readable full results with enriched papers and scores

---

## Quickstart

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) — free tier, no credit card needed.

### 2. Install

```bash
git clone https://github.com/Soumedhik/Research-CV-Ranker.git
cd Research-CV-Ranker
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your Groq API key
```

### 3. Run

```bash
# Basic — rank all CVs in a folder
python rank.py --resumes ./cvs/

# With a role description (enables Fit scoring)
python rank.py --resumes ./cvs/ --job "PhD researcher in NLP / LLMs with strong publication record"

# Show only top 5
python rank.py --resumes ./cvs/ --top 5

# Custom scoring weights (must sum to 1.0)
python rank.py --resumes ./cvs/ --weights "publications=0.5,research=0.3,education=0.1,fit=0.05,trajectory=0.05"

# Skip paper lookup (faster, offline-friendly)
python rank.py --resumes ./cvs/ --no-papers

# Export full results to JSON
python rank.py --resumes ./cvs/ --export results.json

# Verbose mode (see rate-limit waits, retries, second-pass extractions)
python rank.py --resumes ./cvs/ --verbose
```

---

## Architecture

```
                          ┌─────────────────┐
                          │  Resume Files    │  PDF / DOCX / TXT / MD
                          │  (N candidates)  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  ResumeParser   │  pdfplumber → raw text
                          └────────┬────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │         LLMAnalyzer (Groq)          │
                 │  • Structured JSON extraction        │
                 │  • Rate limiter (RPM/TPM/RPD/TPD)   │
                 │  • Retry + model fallback            │
                 │  • Second-pass regex pub recovery    │
                 └─────────────────┬──────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │       PaperAnalyzer (enrichment)    │
                 │  • Semantic Scholar → OpenAlex       │
                 │    → Crossref (cascading fallback)   │
                 │  • VenueRanker (CORE + Scimago       │
                 │    + config + heuristics)             │
                 │  • Paper & venue caching              │
                 └─────────────────┬──────────────────┘
                                   │
                          ┌────────▼────────┐
                          │     Scorer      │  5-dimension weighted score
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ TerminalReporter│  Rich tables, panels, bars
                          │  + JSON export  │
                          └─────────────────┘
```

---

## Scoring System

| Dimension | Default Weight | What it measures |
|-----------|---------------|-----------------|
| **Publications** | 30% | Venue rank (A\*/A/B/C), citation count, first-authorship, paper count |
| **Research** | 25% | Quality & duration of research roles, institution prestige |
| **Education** | 20% | Degree level (PhD > MS > BS), institution tier |
| **Role Fit** | 15% | Keyword alignment with job description + LLM assessment |
| **Trajectory** | 10% | Recency, career progression, awards |

All dimensions score 0–100. Final score is a weighted sum (0–100).

### Publication Scoring Detail

| Factor | Formula | Normalization |
|--------|---------|---------------|
| Venue score | `RANK_POINTS[rank] × (1.5 if first_author else 1.0)` | cap at 50 |
| Citation score | `log1p(citations) × 2` | cap at 30 |
| Count bonus | `min(paper_count / 10, 0.5)` | — |

Where `RANK_POINTS = {A*: 10, A: 6, B: 3, C: 1, ?: 0.5}`

---

## Venue Ranking

Multi-layer venue ranking with 800+ indexed venues:

| Layer | Source | Coverage |
|-------|--------|----------|
| 1. **CORE Rankings** | Bundled JSON from CORE Portal | ~300 CS conferences (A\*–C) |
| 2. **Scimago SJR** | Bundled JSON, quartile-mapped | ~300 journals (Q1=A\*, Q2=A, Q3=B, Q4=C) |
| 3. **Config lists** | `config.py` curated venues | ~100 ML/NLP/Systems venues |
| 4. **Heuristics** | Keyword rules | IEEE Trans→A, Workshop→C, etc. |

**Venue name normalization**: 300+ alias mappings in `data/venue_aliases.json` handle common variants:

| Input | → Canonical |
|-------|------------|
| "Conference on Neural Information Processing Systems" | NeurIPS |
| "Proceedings of the ACL" | ACL |
| "IEEE TPAMI" | TPAMI |
| "PVLDB" | VLDB |

---

## Paper Enrichment Pipeline

Each publication goes through:

1. **Cache check** — skip API if cached from a previous run (30-day TTL)
2. **Semantic Scholar** — title search, fetch citations/venue/fields/OA status
3. **OpenAlex** (fallback) — broader coverage, cited-by count, concepts
4. **Crossref** (fallback) — DOI-focused, container-title for venue

Results are cached to `.cache/papers.json` to avoid redundant calls.

---

## Supported Formats

| Format | Library |
|--------|---------|
| PDF | `pdfplumber` (primary) or `pypdf` (fallback) |
| DOCX / DOC | `python-docx` |
| TXT | built-in |
| Markdown | built-in |

---

## Free APIs Used

| Service | Purpose | Key needed? | Rate limits |
|---------|---------|-------------|-------------|
| [Groq](https://console.groq.com) | LLM structured extraction | ✅ Free signup | 30 RPM, 30K TPM (scout-17b) |
| [Semantic Scholar](https://api.semanticscholar.org) | Paper lookup, citations | ❌ No | ~100 req/5min |
| [OpenAlex](https://openalex.org) | Fallback paper data | ❌ No | ~10 req/sec (polite pool) |
| [Crossref](https://api.crossref.org) | Fallback paper/DOI data | ❌ No | ~50 req/sec (polite pool) |

---

## Configuration

Edit `config.py` to customize:

- **LLM model** — default `llama-4-scout-17b-16e-instruct` (best free-tier throughput)
- **Fallback model** — `llama-3.1-8b-instant` (highest daily request quota)
- **Default weights** — change scoring dimension weights globally
- **Institution tiers** — add your own tier-1/tier-2 universities
- **Venue rankings** — add domain-specific venues (bioinformatics, HCI, etc.)
- **CV truncation** — `groq_cv_truncate_chars` (default 10K chars)

### Venue Data Customization

Add venues to `data/core_rankings.json`, `data/scimago_rankings.json`, or `data/venue_aliases.json` to improve ranking coverage for your domain.

---

## Project Structure

```
├── rank.py                         # CLI entry point (convenience alias)
├── main.py                         # Full CLI with argparse
├── config.py                       # All configuration
├── requirements.txt
├── .env.example                    # Template for API keys
├── LICENSE                         # MIT
│
├── parsers/
│   └── resume_parser.py            # PDF/DOCX/TXT → raw text
│
├── analyzers/
│   ├── llm_analyzer.py             # Groq LLM → structured data + regex second-pass
│   └── paper_analyzer.py           # SS/OpenAlex/Crossref enrichment + venue ranking
│
├── rankers/
│   └── scorer.py                   # Multi-dimensional scoring engine
│
├── reporters/
│   └── terminal_reporter.py        # Rich terminal output + JSON export
│
├── utils/
│   ├── rate_limiter.py             # Proactive Groq rate limiter
│   ├── cache.py                    # File-based paper & venue caching
│   └── venue_data.py               # Loads CORE/Scimago/alias datasets
│
└── data/
    ├── core_rankings.json           # ~300 CORE conference rankings
    ├── scimago_rankings.json        # ~300 Scimago journal quartiles
    └── venue_aliases.json           # ~300 venue name aliases
```

---

## Output Example

```
                    🏆 RESEARCH CANDIDATE LEADERBOARD

  # │ Candidate          │ Total │ Score Bar                     │ Papers │ Rank │ ...
 ───┼────────────────────┼───────┼───────────────────────────────┼────────┼──────┼────
 🥇 │ Jane Smith         │  84.3 │ ████████████████████░░░░░░░░  │   9    │ ★ A* │ ...
 🥈 │ Alex Chen          │  71.8 │ ████████████████░░░░░░░░░░░░  │   5    │ ★ A* │ ...
 🥉 │ Raj Patel          │  68.2 │ ███████████████░░░░░░░░░░░░░  │   7    │ ◆ A  │ ...
```

---

## Notes

- **Rate limiting**: Proactive RPM/TPM/RPD/TPD tracking with 85% safety margin. For large batches (30+ CVs), the tool automatically waits and backs off.
- **Caching**: Paper lookups and venue ranks are cached to `.cache/` (30-day and 90-day TTL respectively). Delete `.cache/` to force fresh lookups.
- **Second-pass extraction**: Regex patterns catch publications the LLM missed from the raw CV text. Deduplication prevents double-counting.
- **Paper lookup** uses cascading fallback: Semantic Scholar → OpenAlex → Crossref. Coverage typically >80% of listed papers.
- This tool is for **lab-internal ranking only** — treat LLM outputs as decision support, not ground truth.

---

## License

MIT — see [LICENSE](LICENSE)
