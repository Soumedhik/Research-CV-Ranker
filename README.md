# 🔬 ResearchRank

> Mass-ranking academic CVs/resumes for research labs — terminal-native, LLM-powered, fully free.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20%28free%29-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

Feed it N resumes (PDF, DOCX, TXT, Markdown). Get back a ranked leaderboard with:

- **Structured extraction** — name, education, research roles, publications, skills, awards
- **Publication enrichment** — looks up each paper on Semantic Scholar (free API), fetches citation counts, open-access status, field of study
- **Venue ranking** — infers conference/journal rank (A\*, A, B, C) from a curated list of 100+ venues (NeurIPS, ICML, Nature, ACL, etc.) + heuristics
- **Publication timeline** — ASCII bar chart showing research activity per year
- **Multi-dimensional scoring** — Publications, Research Experience, Education, Role Fit, Trajectory
- **Rich terminal output** — leaderboard table, per-candidate deep-dive cards, score bars
- **JSON export** — machine-readable full results

---

## Quickstart

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) — free tier, no credit card needed.

```bash
export GROQ_API_KEY=gsk_your_key_here
```

### 2. Install

```bash
git clone https://github.com/your-lab/researchrank
cd researchrank
pip install -r requirements.txt
```

### 3. Run

```bash
# Basic — rank all CVs in a folder
python main.py --resumes ./cvs/

# With a role description (enables Fit scoring)
python main.py --resumes ./cvs/ --job "PhD researcher in NLP / LLMs with strong publication record"

# Show only top 5
python main.py --resumes ./cvs/ --top 5

# Custom scoring weights (must sum to 1.0)
python main.py --resumes ./cvs/ --weights "publications=0.5,research=0.3,education=0.1,fit=0.05,trajectory=0.05"

# Skip paper lookup (faster, offline-friendly)
python main.py --resumes ./cvs/ --no-papers

# Export full results to JSON
python main.py --resumes ./cvs/ --export results.json

# Verbose mode (see each step)
python main.py --resumes ./cvs/ --verbose
```

---

## Scoring System

| Dimension | Default Weight | What it measures |
|-----------|---------------|-----------------|
| **Publications** | 30% | Venue rank (A\*/A/B/C), citation count, first-authorship, paper count |
| **Research** | 25% | Quality & duration of research roles, institution prestige (MIT, DeepMind, etc.) |
| **Education** | 20% | Degree level (PhD > MS > BS), institution tier |
| **Role Fit** | 15% | Keyword alignment with job description + LLM assessment |
| **Trajectory** | 10% | Recency, career progression, awards |

All dimensions score 0–100. Final score is a weighted sum (0–100).

### Venue Rankings

Built from CORE, Scimago, and curated CS/ML conference lists:

| Rank | Examples |
|------|---------|
| **A\*** | NeurIPS, ICML, ICLR, CVPR, ACL, Nature, Science, SIGIR, STOC |
| **A** | AAAI, IJCAI, AISTATS, JMLR, TPAMI, INTERSPEECH, ICSE |
| **B** | ICWSM, CIKM, BMVC, ICDM, WACV |
| **C** | Workshops, arXiv preprints |

---

## Supported Formats

| Format | Library |
|--------|---------|
| PDF | `pdfplumber` (recommended) or `pypdf` |
| DOCX / DOC | `python-docx` |
| TXT | built-in |
| Markdown | built-in |

---

## Free APIs Used

| Service | Purpose | Key needed? |
|---------|---------|------------|
| [Groq](https://console.groq.com) | LLM extraction (`llama-3.3-70b-versatile`) | ✅ Free signup |
| [Semantic Scholar](https://api.semanticscholar.org) | Paper lookup, citations | ❌ No |
| [OpenAlex](https://openalex.org) | Fallback paper data | ❌ No |

---

## Configuration

Edit `config.py` to customize:

- **LLM model** — default `llama-3.3-70b-versatile` (best free quality), fallback `llama3-8b-8192` (faster)
- **Default weights** — change scoring dimension weights globally
- **Institution tiers** — add your own tier-1/tier-2 universities
- **Venue rankings** — add domain-specific venues (bioinformatics, HCI, etc.)

---

## Project Structure

```
researchrank/
├── main.py                     # CLI entrypoint
├── config.py                   # All configuration
├── requirements.txt
├── parsers/
│   └── resume_parser.py        # PDF/DOCX/TXT → raw text
├── analyzers/
│   ├── llm_analyzer.py         # Groq LLM → structured data
│   └── paper_analyzer.py       # Semantic Scholar lookup + venue ranking
├── rankers/
│   └── scorer.py               # Multi-dimensional scoring engine
└── reporters/
    └── terminal_reporter.py    # Rich terminal output + JSON export
```

---

## Output Example

```
                    🏆 RESEARCH CANDIDATE LEADERBOARD

  # │ Candidate          │ Total │ Score Bar                     │ Papers │ Top Venue │ ...
 ───┼────────────────────┼───────┼───────────────────────────────┼────────┼───────────┼────
 🥇 │ Jane Smith         │  84.3 │ ████████████████████░░░░░░░░  │   9    │  ★ A*     │ ...
 🥈 │ Alex Chen          │  71.8 │ ████████████████░░░░░░░░░░░░  │   5    │  ★ A*     │ ...
 🥉 │ Raj Patel          │  68.2 │ ███████████████░░░░░░░░░░░░░  │   7    │  ◆ A      │ ...
```

---

## Notes

- Groq free tier has rate limits (~30 req/min). For large batches (30+ CVs), the tool automatically backs off and retries.
- Paper lookup uses a 1.2s delay between Semantic Scholar requests to stay within free rate limits.
- The tool is for **lab-internal ranking only** — treat LLM outputs as decision support, not ground truth.

---

## License

MIT
