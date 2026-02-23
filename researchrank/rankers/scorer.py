"""
rankers/scorer.py
Multi-dimensional scoring and ranking of analyzed resumes.

Dimensions:
  publications  — count, venue rank, citations, first-authorship
  research      — experience quality, duration, institution quality
  education     — degree level, institution prestige
  fit           — alignment with job description (keyword + semantic)
  trajectory    — recency, progression, growth
"""

import re
import math
from dataclasses import dataclass
from typing import Optional

from config import Config

RANK_POINTS = {"A*": 10, "A": 6, "B": 3, "C": 1, "?": 0.5}


class Scorer:
    def __init__(self, weights: Optional[dict] = None, job_description: Optional[str] = None):
        self.cfg = Config()
        self.weights = weights or self.cfg.default_weights
        self.job_description = job_description
        self.job_keywords = self._extract_keywords(job_description) if job_description else set()

    def rank_all(self, resumes: list) -> list:
        for r in resumes:
            if r.llm_error:
                r.scores = {}
                r.total_score = 0.0
                continue
            r.scores = self._compute_scores(r)
            r.total_score = self._weighted_total(r.scores)

        ranked = sorted(resumes, key=lambda x: x.total_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    def _compute_scores(self, r) -> dict:
        return {
            "publications":  self._score_publications(r),
            "research":      self._score_research(r),
            "education":     self._score_education(r),
            "fit":           self._score_fit(r),
            "trajectory":    self._score_trajectory(r),
        }

    def _weighted_total(self, scores: dict) -> float:
        total = 0.0
        for dim, w in self.weights.items():
            total += scores.get(dim, 0) * w
        return round(total, 2)

    # ── Publications ──────────────────────────────────────────────────────────

    def _score_publications(self, r) -> float:
        papers = r.enriched_papers or []

        if not papers and r.publications:
            # No enrichment — use raw publications with venue-only scoring
            papers = [
                {
                    "venue_rank": self._quick_rank(p.venue),
                    "citation_count": 0,
                    "is_first_author": p.is_first_author,
                    "year": p.year,
                }
                for p in r.publications
            ]

        if not papers:
            return 0.0

        venue_score = 0.0
        citation_score = 0.0
        first_author_bonus = 0.0

        for p in papers:
            rank = p.get("venue_rank", "?")
            pts = RANK_POINTS.get(rank, 0)
            if p.get("is_first_author"):
                pts *= 1.5
                first_author_bonus += 0.5
            venue_score += pts

            cites = p.get("citation_count", 0) or 0
            citation_score += math.log1p(cites) * 2

        # Normalize: venue score caps around 50+ for excellent candidates
        normalized_venue = min(venue_score / 50.0, 1.0)
        normalized_cites = min(citation_score / 30.0, 1.0)
        count_bonus = min(len(papers) / 10.0, 0.5)

        raw = (normalized_venue * 0.55 + normalized_cites * 0.30 + count_bonus * 0.15)
        return round(min(raw, 1.0) * 100, 1)

    def _quick_rank(self, venue: str) -> str:
        from analyzers.paper_analyzer import VenueRanker
        r, _ = VenueRanker().rank(venue)
        return r

    # ── Research Experience ────────────────────────────────────────────────────

    def _score_research(self, r) -> float:
        import datetime
        current_year = datetime.datetime.now().year

        exps = r.research_experience
        if not exps:
            return 0.0

        org_score = 0.0
        duration_years = 0.0

        for exp in exps:
            org_lower = exp.organization.lower()
            # Check tier
            tier = 0
            if any(k in org_lower for k in self.cfg.tier1_keywords):
                tier = 3
            elif any(k in org_lower for k in self.cfg.tier2_keywords):
                tier = 2
            elif exp.is_academic:
                tier = 1

            # Industry research labs
            top_labs = ["deepmind", "google brain", "openai", "fair", "msr",
                        "microsoft research", "ibm research", "amazon science",
                        "meta ai", "apple research", "nvidia research"]
            if any(lab in org_lower for lab in top_labs):
                tier = max(tier, 2)

            org_score += tier

            # Duration
            sy = exp.start_year
            ey = exp.end_year or current_year
            if sy and ey >= sy:
                duration_years += (ey - sy)

        norm_org = min(org_score / 9.0, 1.0)       # 3 tier-3 orgs = max
        norm_dur = min(duration_years / 6.0, 1.0)   # 6 years = max
        count_bonus = min(len(exps) / 5.0, 0.2)

        raw = norm_org * 0.55 + norm_dur * 0.30 + count_bonus * 0.15
        return round(min(raw, 1.0) * 100, 1)

    # ── Education ─────────────────────────────────────────────────────────────

    def _score_education(self, r) -> float:
        if not r.education:
            return 0.0

        degree_map = {"phd": 5, "ph.d": 5, "doctorate": 5,
                      "ms": 3, "msc": 3, "m.s": 3, "master": 3,
                      "meng": 2.5, "m.eng": 2.5,
                      "bs": 2, "b.s": 2, "ba": 1.5, "b.a": 1.5,
                      "be": 2, "b.e": 2, "btech": 2, "b.tech": 2}

        best_degree_score = 0
        best_inst_score = 0

        for edu in r.education:
            degree_lower = edu.degree.lower()
            d_score = 0
            for key, val in degree_map.items():
                if key in degree_lower:
                    d_score = val
                    break
            best_degree_score = max(best_degree_score, d_score)

            inst_lower = edu.institution.lower()
            i_score = 0
            if any(k in inst_lower for k in self.cfg.tier1_keywords):
                i_score = 3
            elif any(k in inst_lower for k in self.cfg.tier2_keywords):
                i_score = 2
            else:
                i_score = 1
            best_inst_score = max(best_inst_score, i_score)

        norm_degree = min(best_degree_score / 5.0, 1.0)
        norm_inst = min(best_inst_score / 3.0, 1.0)

        raw = norm_degree * 0.5 + norm_inst * 0.5
        return round(min(raw, 1.0) * 100, 1)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def _score_fit(self, r) -> float:
        if not self.job_keywords:
            return 50.0   # Neutral when no job description

        candidate_text = " ".join([
            r.research_summary,
            " ".join(r.skills),
            " ".join(e.description for e in r.research_experience),
            " ".join(p.venue for p in r.publications),
        ]).lower()

        matched = sum(1 for kw in self.job_keywords if kw in candidate_text)
        keyword_ratio = matched / len(self.job_keywords)

        # fit_notes from LLM boosts score
        if r.fit_notes:
            positive_signals = ["strong fit", "excellent", "highly relevant",
                                 "ideal", "well-suited", "directly relevant"]
            negative_signals = ["weak fit", "limited", "not aligned", "lacks",
                                 "no experience", "insufficient"]
            fit_lower = r.fit_notes.lower()
            llm_boost = sum(0.1 for s in positive_signals if s in fit_lower)
            llm_penalty = sum(0.1 for s in negative_signals if s in fit_lower)
            llm_modifier = min(max(llm_boost - llm_penalty, -0.3), 0.3)
        else:
            llm_modifier = 0.0

        raw = keyword_ratio + llm_modifier
        return round(min(max(raw, 0.0), 1.0) * 100, 1)

    # ── Trajectory ────────────────────────────────────────────────────────────

    def _score_trajectory(self, r) -> float:
        import datetime
        current_year = datetime.datetime.now().year

        scores = []

        # Recency of publications
        years = [p.year for p in r.publications if p.year]
        if years:
            most_recent = max(years)
            recency = 1.0 - min((current_year - most_recent) / 5.0, 1.0)
            scores.append(recency)

        # Career progression (PhD or research roles)
        has_phd = any("phd" in e.degree.lower() or "ph.d" in e.degree.lower()
                      for e in r.education)
        if has_phd:
            scores.append(0.8)

        # Growing number of roles / experiences
        if len(r.research_experience) > 1:
            scores.append(min(len(r.research_experience) / 4.0, 1.0))

        # Awards
        if r.awards:
            scores.append(min(len(r.awards) / 3.0, 1.0))

        if not scores:
            return 30.0

        return round(sum(scores) / len(scores) * 100, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> set:
        if not text:
            return set()
        stop = {"the", "and", "or", "for", "with", "in", "of", "a", "an",
                 "to", "is", "are", "we", "our", "that", "this", "on", "at"}
        words = re.findall(r"\b[a-z][a-z\-]{2,}\b", text.lower())
        return {w for w in words if w not in stop}
