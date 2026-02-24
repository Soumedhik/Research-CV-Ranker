# ResearchRank

> **Mass-rank academic CVs for your lab — terminal-native, free-tier LLM, publication-aware.**
>
> Built for professors and lab directors who need a fast, rigorous shortlist from a large applicant pool.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20free%20tier-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![APIs](https://img.shields.io/badge/APIs-Semantic%20Scholar%20%7C%20OpenAlex%20%7C%20Crossref-green)

---

## What it does

Drop N CVs (PDF / DOCX / TXT / Markdown) into a folder. Run one command.  
Get back a fully ranked, annotated leaderboard — with **real publication metrics** pulled live from Semantic Scholar.

```
                       ── Pipeline ──
  PDF/DOCX/TXT   ──▶  ResumeParser  ──▶  Groq LLM (structured extraction)
                                    ──▶  PaperAnalyzer (SS → OpenAlex → Crossref)
                                    ──▶  Scorer (H-index · citations · venue tiers)
                                    ──▶  TerminalReporter (Rich UI + JSON export)
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **LLM extraction** | Groq (free tier) extracts name, education, research roles, publications, skills, awards |
| **Second-pass recovery** | Regex catches papers the LLM missed; deduplication by title similarity |
| **Paper enrichment** | Cascading lookup: Semantic Scholar → OpenAlex → Crossref; cached with 30-day TTL |
| **Venue ranking** | 800+ venues from bundled **CORE** + **Scimago SJR** + curated config lists + heuristics |
| **H-index estimation** | Computed from real citation counts fetched for each paper |
| **Academic metrics** | Total citations, avg citations/paper, years active, venue tier breakdown (A\*/A/B/C/?) |
| **Multi-dim scoring** | Publications 30% · Research 25% · Education 20% · Role Fit 15% · Trajectory 10% |
| **Rich terminal UI** | Leaderboard with `█/░` score bars · per-candidate profile panels · SWOT · timeline |
| **Rate limiting** | Proactive RPM/TPM/RPD/TPD tracking — never hits Groq 429 errors |
| **JSON export** | Full results including all metrics, enriched papers, and scores |

---

## Quickstart

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) — free, no credit card needed.

### 2. Install

```bash
git clone https://github.com/Soumedhik/Research-CV-Ranker.git
cd Research-CV-Ranker
pip install -r requirements.txt
cp .env.example .env
# paste your GROQ_API_KEY into .env
```

### 3. Run

```bash
# Rank all CVs in a folder
python main.py --resumes ./cvs/

# With a role description (enables Role Fit scoring)
python main.py --resumes ./cvs/ --job "Postdoc in NLP / LLMs with strong publication record"

# Show only top 5
python main.py --resumes ./cvs/ --top 5

# Export full results to JSON
python main.py --resumes ./cvs/ --job "ML researcher" --export results.json

# Custom scoring weights (must sum to 1.0)
python main.py --resumes ./cvs/ --weights "publications=0.5,research=0.3,education=0.1,fit=0.05,trajectory=0.05"

# Skip paper lookup (faster, offline-friendly)
python main.py --resumes ./cvs/ --no-papers

# Verbose (see rate-limit waits, retries, second-pass extractions)
python main.py --resumes ./cvs/ --verbose
```

`rank.py` is a convenience alias — `python rank.py ...` is identical to `python main.py ...`.

---

## Setup: `.env` file

```dotenv
# .env  (never commit this file)
GROQ_API_KEY=gsk_your_key_here
```

All other services (Semantic Scholar, OpenAlex, Crossref) are free with no key required.

---

## Output: What you see

### Leaderboard (top of output)

```
          🏆  RESEARCH CANDIDATE LEADERBOARD  ·  Role: Postdoc in NLP

  #  │ Candidate          │ Total │ Score Bar               │ Papers │ H-idx │  Cites │ Top Tier │ P·R·E·F·T
 ────┼────────────────────┼───────┼─────────────────────────┼────────┼───────┼────────┼──────────┼──────────
 🥇  │ Jane Smith         │  84.3 │ ████████████████████░░░ │   9    │   5   │  1,203 │  ★ A*    │ 95·80·72·90·85
 🥈  │ Alex Chen          │  71.8 │ ████████████████░░░░░░░ │   5    │   3   │    412 │  ★ A*    │ 70·65·75·80·70
 🥉  │ Raj Patel          │  68.2 │ ███████████████░░░░░░░░ │   7    │   2   │    188 │  ◆ A     │ 65·72·68·65·72
```

### Per-candidate profile (scrollable)

Each candidate gets a deep-dive panel:

- **Score Breakdown** — `█/░` bars for each of the 5 dimensions
- **Academic Impact Metrics** — H-index · Total citations · Avg citations/paper · Years active · Tier breakdown (A\*:N | A:N | B:N | C:N | ?:N) · First-author % · H-bar visual
- **Research Summary** — LLM-generated narrative
- **Education** table
- **Research Experience** list (academic vs industry colour-coded)
- **Publications** table with tier badge, citations, influential citations, first-author ✓, source
- **Publication Timeline** — `█` bars per year, colour-coded by best venue tier
- **Skills & Awards** panels
- **Role Fit Assessment** — LLM evaluation against provided job description
- **SWOT** — Strengths and Weaknesses panels

### Cohort Summary (end of output)

```
  Candidates processed     35
  Total publications        53
  Total citations (cohort)  15,822
  A* / top-tier papers      16
  Highest H-index           4
  Average total score       30.6
  Top candidate             Jane Smith (84.3)
```

---

## Scoring System

### Five Dimensions

| Dimension | Default Weight | What it measures |
|-----------|---------------|-----------------|
| **Publications** | 30% | Venue rank (A\*/A/B/C), citation count, first-authorship, paper count |
| **Research** | 25% | Quality & duration of research roles, institution prestige |
| **Education** | 20% | Degree level (PhD > MS > BS), institution tier |
| **Role Fit** | 15% | Keyword and semantic alignment with `--job` description |
| **Trajectory** | 10% | Recency, career progression, awards |

All scores are 0–100; final score is a weighted sum.

### Academic Metrics (computed before scoring)

For every candidate, before any dimension score is calculated:

```
h_index          — max h where h papers have ≥ h citations
total_citations  — sum of all citation counts
avg_citations    — total_citations / paper_count
years_active     — max(pub_years) - min(pub_years) + 1
venue_tier_counts — {"A*": N, "A": N, "B": N, "C": N, "?": N}
```

These feed into the Publications score and are also displayed directly in the UI and JSON export.

---

## Venue Ranking

800+ indexed venues across four layers:

| Layer | Source | Coverage |
|-------|--------|----------|
| 1. **CORE Rankings** | Bundled JSON from CORE Portal | ~300 CS conferences (A\*–C) |
| 2. **Scimago SJR** | Bundled JSON, quartile-mapped (Q1=A\*, Q2=A, Q3=B, Q4=C) | ~300 journals |
| 3. **Config lists** | `config.py` curated venues (NeurIPS, ACL, ICML, CVPR, …) | ~100 ML/NLP/Systems |
| 4. **Heuristics** | Keyword rules (IEEE Trans→A, Workshop→C, …) | fallback |

**300+ venue aliases** (`data/venue_aliases.json`) handle variant names, e.g.:
- "Conference on Neural Information Processing Systems" → NeurIPS
- "Proceedings of the ACL" → ACL  
- "IEEE TPAMI" → TPAMI

---

## Paper Enrichment Pipeline

```
For each publication title:
  1. Cache check (.cache/papers.json, 30-day TTL)  → hit: done
  2. Semantic Scholar title search                  → found: enrich
  3. OpenAlex title search (fallback)               → found: enrich
  4. Crossref title search (fallback)               → found: enrich
  5. Not found: mark as "?" tier, 0 citations
```

Results include: `citation_count`, `influential_citation_count`, `venue_canonical`, `venue_rank`, `year`, `is_first_author`, `lookup_source`.

---

## Project Structure

```
├── main.py                         # CLI entry point (argparse)
├── rank.py                         # Convenience alias for main.py
├── config.py                       # All configuration (reads from .env)
├── requirements.txt
├── .env.example                    # Template — copy to .env, add GROQ_API_KEY
├── LICENSE
│
├── parsers/
│   └── resume_parser.py            # PDF/DOCX/TXT/MD → raw text
│
├── analyzers/
│   ├── llm_analyzer.py             # Groq LLM structured extraction + regex second-pass
│   └── paper_analyzer.py           # SS/OpenAlex/Crossref enrichment + venue ranking
│
├── rankers/
│   └── scorer.py                   # H-index + 5-dimension weighted scoring engine
│
├── reporters/
│   └── terminal_reporter.py        # Rich terminal UI (leaderboard, profiles) + JSON export
│
├── utils/
│   ├── rate_limiter.py             # Proactive Groq RPM/TPM/RPD/TPD rate limiter
│   ├── cache.py                    # File-based paper & venue caching
│   └── venue_data.py               # Loads CORE/Scimago/alias datasets
│
└── data/
    ├── core_rankings.json           # CORE conference rankings
    ├── scimago_rankings.json        # Scimago journal quartiles
    └── venue_aliases.json           # Venue name aliases
```

---

## Free APIs Used

| Service | Purpose | Key needed? | Rate limits |
|---------|---------|-------------|-------------|
| [Groq](https://console.groq.com) | LLM structured extraction | ✅ Free signup | 30 RPM, 30K TPM (scout-17b) |
| [Semantic Scholar](https://api.semanticscholar.org) | Paper lookup, citations | ❌ No key | ~100 req/5min |
| [OpenAlex](https://openalex.org) | Fallback paper data | ❌ No key | ~10 req/sec (polite pool) |
| [Crossref](https://api.crossref.org) | Fallback paper/DOI data | ❌ No key | ~50 req/sec (polite pool) |

---

## Configuration

Edit `config.py` or override via environment variables:

- **LLM model** — default `meta-llama/llama-4-scout-17b-16e-instruct` (best free-tier throughput)
- **Fallback model** — `llama-3.1-8b-instant` (highest daily request quota)
- **Scoring weights** — change per-dimension weights globally
- **Institution tiers** — add your own tier-1/tier-2 universities
- **Venue rankings** — add domain-specific venues (bioinformatics, HCI, etc.)

---

## Notes

- **Rate limiting** — Proactive tracking with 85% safety margin. Large batches (30+ CVs) auto-wait and back off. No manual intervention needed.
- **Caching** — `.cache/papers.json` (30-day TTL), `.cache/venues.json` (90-day TTL). Delete `.cache/` to force fresh lookups.
- **Security** — Resume PDFs, `.env`, and all result files are gitignored. Never commit candidate data.
- **Lab use only** — Treat LLM outputs as decision support, not ground truth. Manual review of shortlisted candidates is strongly recommended.

---

## License

MIT — see [LICENSE](LICENSE)
