# -*- coding: utf-8 -*-
"""
reporters/terminal_reporter.py

Rich-powered terminal output for the ResearchRank framework.
Designed for academic evaluators (lab directors / professors).

Features:
  - Leaderboard with H-index, citations, top venue tier
  - Per-candidate deep academic metrics panel
  - Publication table with tier badges + citation counts
  - Unicode score bars and publication timeline (colour-coded by tier)
  - SWOT + job-fit panels
  - Groq API usage summary
  - JSON export
"""

import json
import math
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.padding import Padding

RANK_COLORS = {
    "A*": "bold magenta",
    "A":  "bold cyan",
    "B":  "bold yellow",
    "C":  "dim yellow",
    "?":  "dim white",
}
RANK_BADGES = {
    "A*": "★ A*",
    "A":  "◆ A ",
    "B":  "● B ",
    "C":  "○ C ",
    "?":  "· ? ",
}
TIMELINE_COLORS = {
    "A*": "magenta",
    "A":  "cyan",
    "B":  "yellow",
    "C":  "white",
    "?":  "dim",
}
MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}

SCORE_BAR_WIDTH = 28
FILL_CHAR  = "█"
EMPTY_CHAR = "░"


def score_bar(score: float, width: int = SCORE_BAR_WIDTH) -> str:
    filled = max(0, min(width, int(round((score / 100.0) * width))))
    empty  = width - filled
    if score >= 75:
        color = "green"
    elif score >= 50:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{FILL_CHAR * filled}[/{color}][dim]{EMPTY_CHAR * empty}[/dim]"


def score_color(score: float) -> str:
    if score >= 75:
        return "bold green"
    elif score >= 50:
        return "bold yellow"
    return "bold red"


def _tier_badge(tier: str) -> str:
    return RANK_BADGES.get(tier, tier)


def _best_rank(papers: list) -> str:
    priority = {"A*": 0, "A": 1, "B": 2, "C": 3, "?": 4}
    ranks = [p.get("venue_rank", "?") for p in papers] if papers else ["?"]
    return min(ranks, key=lambda x: priority.get(x, 4))


class TerminalReporter:
    def __init__(self, console: Console):
        self.console = console

    def render(self, ranked: list, top_n: Optional[int] = None,
               job_description: Optional[str] = None):
        display = ranked[:top_n] if top_n else ranked
        self._render_leaderboard(display, job_description)
        self.console.print()
        for r in display:
            if r.llm_error:
                self._render_error_card(r)
            else:
                self._render_candidate_profile(r)
        self._render_summary_stats(ranked)

    def _render_leaderboard(self, ranked: list, job_description: Optional[str]):
        title = "🏆  RESEARCH CANDIDATE LEADERBOARD"
        if job_description:
            title += f"  ·  [dim]Role: {job_description[:60]}[/dim]"

        table = Table(
            title=title,
            box=box.HEAVY_HEAD,
            border_style="cyan",
            header_style="bold white on dark_blue",
            show_lines=True,
            padding=(0, 1),
            min_width=110,
        )
        table.add_column("#",        style="bold",   width=4,  justify="center")
        table.add_column("Candidate",               min_width=22)
        table.add_column("Total",    width=8,        justify="center")
        table.add_column("Score Bar",               min_width=30)
        table.add_column("Papers",   width=8,        justify="center")
        table.add_column("H-idx",    width=6,        justify="center")
        table.add_column("Cites",    width=8,        justify="right")
        table.add_column("Top Tier", width=9,        justify="center")
        table.add_column("Pubs·Res·Edu·Fit·Traj",  min_width=26)

        for r in ranked:
            medal = MEDAL.get(r.rank, f"  {r.rank}")
            name_txt = Text(r.name or "Unknown", style="bold white")
            if r.email:
                name_txt.append(f"\n{r.email}", style="dim")

            papers  = r.enriched_papers or []
            n_pubs  = len(papers) or len(r.publications or [])
            top_tier = _best_rank(papers) if papers else "?"
            tier_txt = Text(_tier_badge(top_tier), style=RANK_COLORS.get(top_tier, "white"))

            s = r.scores
            score_mini = (
                f"[cyan]{s.get('publications', 0):.0f}[/cyan]"
                f"·[blue]{s.get('research', 0):.0f}[/blue]"
                f"·[green]{s.get('education', 0):.0f}[/green]"
                f"·[yellow]{s.get('fit', 0):.0f}[/yellow]"
                f"·[magenta]{s.get('trajectory', 0):.0f}[/magenta]"
            )

            if r.llm_error:
                table.add_row(
                    medal, Text(r.name or "?", style="dim red"),
                    "ERR", "—", "—", "—", "—", "—", "—",
                )
            else:
                table.add_row(
                    medal,
                    name_txt,
                    Text(f"{r.total_score:.1f}", style=score_color(r.total_score)),
                    score_bar(r.total_score),
                    Text(str(n_pubs), style="cyan"),
                    Text(str(getattr(r, "h_index", 0)), style="bold green"),
                    Text(str(getattr(r, "total_citations", 0)), style="bold cyan"),
                    tier_txt,
                    score_mini,
                )

        self.console.print(table)
        self.console.print(
            "[dim]Scores: [cyan]Pubs[/cyan]·[blue]Research[/blue]·"
            "[green]Education[/green]·[yellow]Fit[/yellow]·[magenta]Trajectory[/magenta]  "
            "| Tiers: [bold magenta]A*[/bold magenta]≥"
            "[bold cyan]A[/bold cyan]>[bold yellow]B[/bold yellow]>"
            "[dim yellow]C[/dim yellow][/dim]"
        )

    def _render_candidate_profile(self, r):
        medal = MEDAL.get(r.rank, f"#{r.rank}")
        header_parts = [f"{medal}  {r.name}"]
        if r.email:
            header_parts.append(r.email)
        if r.homepage:
            header_parts.append(r.homepage)
        self.console.print(
            Rule(f"[bold cyan]{' · '.join(header_parts)}[/bold cyan]", style="cyan")
        )
        self._render_score_breakdown(r)
        self._render_academic_metrics(r)

        if r.research_summary:
            self.console.print(Panel(
                f"[white]{r.research_summary}[/white]",
                title="[bold]Research Summary[/bold]",
                border_style="blue",
                padding=(0, 2),
            ))

        if r.education:
            self._render_education(r)
        if r.research_experience:
            self._render_research_experience(r)
        if r.publications or r.enriched_papers:
            self._render_publications(r)

        row_items = []
        if r.skills:
            row_items.append(Panel(
                f"[cyan]{', '.join(r.skills[:18])}[/cyan]",
                title="[bold]Skills[/bold]",
                border_style="dim cyan", padding=(0, 1),
            ))
        if r.awards:
            row_items.append(Panel(
                "\n".join(f"🏅 {a}" for a in r.awards[:5]),
                title="[bold]Awards[/bold]",
                border_style="dim yellow", padding=(0, 1),
            ))
        if row_items:
            self.console.print(Columns(row_items, equal=False, expand=True))

        if r.fit_notes:
            self.console.print(Panel(
                f"[italic yellow]{r.fit_notes}[/italic yellow]",
                title="[bold yellow]Role Fit Assessment[/bold yellow]",
                border_style="yellow", padding=(0, 2),
            ))

        if r.strengths or r.weaknesses:
            self._render_swot(r)

        self.console.print()

    def _render_score_breakdown(self, r):
        dims = [
            ("publications", "cyan",    "Publications"),
            ("research",     "blue",    "Research Exp"),
            ("education",    "green",   "Education"),
            ("fit",          "yellow",  "Role Fit"),
            ("trajectory",   "magenta", "Trajectory"),
        ]
        lines = []
        for key, color, label in dims:
            sc = r.scores.get(key, 0)
            lines.append(
                f"  [{color}]{label:<16}[/{color}] {score_bar(sc, 22)} "
                f"[{score_color(sc)}]{sc:5.1f}[/{score_color(sc)}]"
            )
        lines.append(
            f"  [bold white]{'TOTAL':<16}[/bold white] {score_bar(r.total_score, 22)} "
            f"[bold white]{r.total_score:5.1f}[/bold white]"
        )
        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Score Breakdown[/bold]",
            border_style="cyan",
            padding=(0, 1),
        ))

    def _render_academic_metrics(self, r):
        h_idx     = getattr(r, "h_index", 0)
        total_c   = getattr(r, "total_citations", 0)
        avg_c     = getattr(r, "avg_citations", 0.0)
        yrs       = getattr(r, "years_active", 0)
        tc_counts = getattr(r, "venue_tier_counts", {})
        papers    = r.enriched_papers or r.publications or []
        n_pubs    = len(papers)

        metrics_line = (
            f"  [bold green]H-index:[/bold green] [bold white]{h_idx}[/bold white]"
            f"   [bold cyan]Total Citations:[/bold cyan] [bold white]{total_c:,}[/bold white]"
            f"   [bold yellow]Avg/Paper:[/bold yellow] [bold white]{avg_c:.1f}[/bold white]"
            f"   [bold blue]Active Years:[/bold blue] [bold white]{yrs}[/bold white]"
            f"   [bold magenta]Papers:[/bold magenta] [bold white]{n_pubs}[/bold white]"
        )

        tier_parts = []
        for tier in ("A*", "A", "B", "C", "?"):
            cnt = tc_counts.get(tier, 0)
            color = RANK_COLORS.get(tier, "white")
            tier_parts.append(f"[{color}]{tier}:{cnt}[/{color}]")
        tier_line = "  Venue tiers:  " + "  |  ".join(tier_parts)

        if r.enriched_papers:
            fa_count = sum(1 for p in r.enriched_papers if p.get("is_first_author"))
        else:
            fa_count = sum(1 for p in (r.publications or []) if p.is_first_author)
        fa_line = (
            f"  First-author papers: [bold white]{fa_count}[/bold white]"
            f"  ({int(fa_count/max(n_pubs,1)*100)}%)"
        )

        h_bar_len = min(h_idx, 20)
        h_bar = (
            f"  H-bar: [bold green]{FILL_CHAR * h_bar_len}[/bold green]"
            f"[dim]{EMPTY_CHAR * (20 - h_bar_len)}[/dim]"
            f"  [dim](max=20)[/dim]"
        )

        content = "\n".join([metrics_line, "", tier_line, fa_line, h_bar])
        self.console.print(Panel(
            content,
            title="[bold green]Academic Impact Metrics[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))

    def _render_education(self, r):
        table = Table(box=box.SIMPLE, show_header=True,
                      header_style="bold dim", border_style="dim", padding=(0, 2))
        table.add_column("Degree")
        table.add_column("Field")
        table.add_column("Institution")
        table.add_column("Year", justify="right")
        table.add_column("GPA",  justify="right")
        for e in r.education:
            table.add_row(
                Text(e.degree, style="bold"),
                e.field or "—",
                e.institution or "—",
                str(e.year) if e.year else "—",
                e.gpa or "—",
            )
        self.console.print(Panel(table, title="[bold]Education[/bold]",
                                 border_style="green", padding=(0, 0)))

    def _render_research_experience(self, r):
        items = []
        for exp in sorted(r.research_experience,
                          key=lambda x: x.start_year or 0, reverse=True):
            color = "blue" if exp.is_academic else "green"
            tag   = "[ACad]" if exp.is_academic else "[Indus]"
            line  = Text()
            line.append(f"{tag} ", style=f"dim {color}")
            line.append(exp.role or "Researcher", style="bold white")
            line.append(f" @ {exp.organization}", style=f"bold {color}")
            line.append(f"  [{exp.duration}]", style="dim")
            if exp.description:
                line.append(f"\n   {exp.description[:120]}", style="dim white")
            items.append(line)
        self.console.print(Panel(
            Text("\n\n").join(items) if items else Text("None"),
            title="[bold]Research Experience[/bold]",
            border_style="blue", padding=(0, 1),
        ))

    def _render_publications(self, r):
        papers   = r.enriched_papers or []
        raw_pubs = r.publications or []

        year_map: dict[int, list] = {}
        if papers:
            for p in papers:
                yr = p.get("year") or 0
                year_map.setdefault(yr, []).append(p)
        else:
            for p in raw_pubs:
                yr = p.year or 0
                year_map.setdefault(yr, []).append({
                    "title": p.title, "venue": p.venue,
                    "venue_rank": "?", "citation_count": 0,
                    "is_first_author": p.is_first_author,
                })

        all_papers = sorted(
            (p for yr, ps in year_map.items() for p in ps),
            key=lambda x: x.get("year") or 0, reverse=True,
        )

        table = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold dim", padding=(0, 1))
        table.add_column("Year",  width=6,  justify="center")
        table.add_column("Tier",  width=8,  justify="center")
        table.add_column("Title", min_width=36)
        table.add_column("Venue", min_width=12)
        table.add_column("1st",   width=4,  justify="center")
        table.add_column("Cites", width=7,  justify="right")
        table.add_column("iCit",  width=5,  justify="right")
        table.add_column("Src",   width=4,  justify="center")

        src_map = {
            "semantic_scholar": "SS",
            "openalex":         "OA",
            "crossref":         "CR",
            "cached":           "$$",
        }

        for p in all_papers:
            tier      = p.get("venue_rank", "?")
            t_style   = RANK_COLORS.get(tier, "white")
            badge     = _tier_badge(tier)
            title     = (p.get("title") or "Unknown")[:65]
            venue_raw = p.get("venue_canonical") or p.get("venue") or "—"
            venue_d   = venue_raw[:22]
            cites     = p.get("citation_count", 0) or 0
            icites    = p.get("influential_citation_count", 0) or 0
            first     = "✓" if p.get("is_first_author") else ""
            src_lbl   = src_map.get(p.get("lookup_source", ""), "—")

            table.add_row(
                str(p.get("year") or "?"),
                Text(badge, style=t_style),
                Text(title, style="white"),
                Text(venue_d, style="dim"),
                Text(first, style="bold green"),
                Text(str(cites) if cites else "—",  style="cyan"),
                Text(str(icites) if icites else "—", style="dim cyan"),
                Text(src_lbl, style="dim"),
            )

        self.console.print(Panel(
            table,
            title=f"[bold]Publications ({len(all_papers)})[/bold]  "
                  f"[dim]Cites=total  iCit=influential[/dim]",
            border_style="magenta", padding=(0, 0),
        ))

        timeline = self._build_timeline(year_map)
        if timeline:
            self.console.print(Panel(
                timeline,
                title="[bold]Publication Timeline[/bold]",
                border_style="dim magenta", padding=(0, 1),
            ))

    def _build_timeline(self, year_map: dict) -> Optional[Text]:
        valid_years = sorted(yr for yr in year_map if 1990 < yr <= 2030)
        if not valid_years:
            return None

        min_y, max_y = min(valid_years), max(valid_years)
        priority = {"A*": 0, "A": 1, "B": 2, "C": 3, "?": 4}
        lines = []

        for yr in range(min_y, max_y + 1):
            ps = year_map.get(yr, [])
            n  = len(ps)
            line = Text(f"  {yr}  ", style="dim white")
            if n == 0:
                line.append("·", style="dim")
            else:
                best = min(
                    (p.get("venue_rank", "?") for p in ps),
                    key=lambda x: priority.get(x, 4),
                )
                color  = TIMELINE_COLORS[best]
                bar_w  = min(n * 4, 32)
                line.append(FILL_CHAR * bar_w, style=color)
                line.append(f"  {n} paper{'s' if n != 1 else ''}  "
                             f"[best: {best}]", style=f"dim {color}")
            lines.append(line)

        return Text("\n").join(lines)

    def _render_swot(self, r):
        items = []
        if r.strengths:
            items.append(Panel(
                "\n".join(f"[green]▲[/green] {s}" for s in r.strengths),
                title="[bold green]Strengths[/bold green]",
                border_style="green", padding=(0, 1),
            ))
        if r.weaknesses:
            items.append(Panel(
                "\n".join(f"[red]▼[/red] {w}" for w in r.weaknesses),
                title="[bold red]Weaknesses[/bold red]",
                border_style="red", padding=(0, 1),
            ))
        if items:
            self.console.print(Columns(items, equal=True, expand=True))

    def _render_error_card(self, r):
        self.console.print(Panel(
            f"[red]Error:[/red] {r.llm_error}",
            title=f"[red]✗ {r.name}[/red]",
            border_style="red",
        ))

    def _render_summary_stats(self, ranked: list):
        valid = [r for r in ranked if not r.llm_error]
        if not valid:
            return

        self.console.print(Rule("[bold]Cohort Summary[/bold]", style="dim"))

        total_papers  = sum(len(r.publications) for r in valid)
        total_cites   = sum(getattr(r, "total_citations", 0) for r in valid)
        astar_papers  = sum(
            1 for r in valid
            for p in r.enriched_papers
            if p.get("venue_rank") == "A*"
        )
        avg_score     = sum(r.total_score for r in valid) / len(valid)
        max_h         = max((getattr(r, "h_index", 0) for r in valid), default=0)

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        table.add_column("Metric", style="dim")
        table.add_column("Value",  style="bold white")

        table.add_row("Candidates processed",    str(len(valid)))
        table.add_row("Total publications",       str(total_papers))
        table.add_row("Total citations (cohort)", f"[cyan]{total_cites:,}[/cyan]")
        table.add_row("A* / top-tier papers",     f"[magenta]{astar_papers}[/magenta]")
        table.add_row("Highest H-index",          f"[green]{max_h}[/green]")
        table.add_row("Average total score",      f"{avg_score:.1f}")
        table.add_row("Top candidate",
                      f"[bold cyan]{valid[0].name}[/bold cyan] "
                      f"({valid[0].total_score:.1f})" if valid else "—")

        self.console.print(Padding(table, (1, 4)))

    def render_api_usage(self, usage: dict):
        self.console.print(Rule("[bold]Groq API Usage[/bold]", style="dim"))

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        table.add_column("Metric", style="dim")
        table.add_column("Value",  style="bold white")

        tokens_used  = usage.get("tokens_used", 0)
        tokens_daily = usage.get("tokens_limit_daily", 1)
        pct = (tokens_used / tokens_daily * 100) if tokens_daily else 0

        table.add_row("Model",       f"[cyan]{usage.get('model', '?')}[/cyan]")
        table.add_row("API calls",   str(usage.get("requests_made", 0)))
        table.add_row("Remaining (daily)",
                      f"{usage.get('requests_remaining', '?'):,} / "
                      f"{usage.get('requests_limit_daily', '?'):,}")
        table.add_row("Tokens used",
                      f"{tokens_used:,}  [dim]({pct:.1f}% of {tokens_daily:,})[/dim]")
        table.add_row("  Prompt",     f"{usage.get('prompt_tokens', 0):,}")
        table.add_row("  Completion", f"{usage.get('completion_tokens', 0):,}")
        table.add_row("Avg/call",     f"{usage.get('avg_tokens_per_call', 0):,}")
        table.add_row("Rate-limit waits",
                      f"{usage.get('rate_limit_waits', 0)}  "
                      f"[dim]({usage.get('wait_seconds', 0):.1f}s)[/dim]")
        table.add_row("Retries",      str(usage.get("retries", 0)))
        table.add_row("Limits",
                      f"[dim]RPM={usage.get('rpm','?')}  "
                      f"TPM={usage.get('tpm','?'):,}[/dim]")

        self.console.print(Padding(table, (1, 4)))

    def export_json(self, ranked: list, path: str):
        output = []
        for r in ranked:
            output.append({
                "rank":                 r.rank,
                "name":                 r.name,
                "email":                r.email,
                "homepage":             r.homepage,
                "total_score":          r.total_score,
                "scores":               r.scores,
                "h_index":              getattr(r, "h_index", 0),
                "total_citations":      getattr(r, "total_citations", 0),
                "avg_citations":        getattr(r, "avg_citations", 0.0),
                "years_active":         getattr(r, "years_active", 0),
                "venue_tier_counts":    getattr(r, "venue_tier_counts", {}),
                "research_summary":     r.research_summary,
                "fit_notes":            r.fit_notes,
                "strengths":            r.strengths,
                "weaknesses":           r.weaknesses,
                "education":            [e.__dict__ for e in r.education],
                "research_experience":  [e.__dict__ for e in r.research_experience],
                "publications_raw":     [p.__dict__ for p in r.publications],
                "publications_enriched": r.enriched_papers,
                "skills":               r.skills,
                "awards":               r.awards,
                "source_file":          str(r.raw.path),
                "error":                r.llm_error,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)