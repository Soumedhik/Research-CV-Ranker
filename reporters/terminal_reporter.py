"""
reporters/terminal_reporter.py
Rich-powered terminal output: ranked leaderboard, per-candidate profiles,
publication timelines, score breakdowns.
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
from rich.bar import Bar
from rich import box
from rich.layout import Layout
from rich.padding import Padding


RANK_COLORS = {"A*": "bold magenta", "A": "bold cyan", "B": "bold yellow", "C": "dim yellow", "?": "dim white"}
RANK_BADGES = {"A*": "★ A*", "A": "◆ A ", "B": "● B ", "C": "○ C ", "?": "· ? "}

MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}
SCORE_BAR_WIDTH = 30


def score_bar(score: float, width: int = SCORE_BAR_WIDTH) -> str:
    """Build a visual bar for a 0–100 score."""
    filled = int((score / 100.0) * width)
    empty = width - filled
    if score >= 75:
        color = "green"
    elif score >= 50:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def score_color(score: float) -> str:
    if score >= 75:
        return "bold green"
    elif score >= 50:
        return "bold yellow"
    else:
        return "bold red"


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

    # ── Leaderboard ───────────────────────────────────────────────────────────

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
            min_width=100,
        )
        table.add_column("#", style="bold", width=4, justify="center")
        table.add_column("Candidate", min_width=22)
        table.add_column("Total", width=8, justify="center")
        table.add_column("Score Bar", min_width=32)
        table.add_column("Papers", width=7, justify="center")
        table.add_column("Top Venue", width=6, justify="center")
        table.add_column("Education", min_width=18)
        table.add_column("Pubs·Res·Edu·Fit·Traj", min_width=25)

        for r in ranked:
            rank_str = MEDAL.get(r.rank, f"  {r.rank}")

            name = Text(r.name or "Unknown", style="bold white")
            if r.email:
                name.append(f"\n{r.email}", style="dim")

            total_text = Text(f"{r.total_score:.1f}", style=score_color(r.total_score))

            bar = score_bar(r.total_score)

            # Publication stats
            papers = r.enriched_papers or []
            raw_pubs = r.publications or []
            paper_count = len(papers) or len(raw_pubs)
            papers_text = Text(str(paper_count), style="cyan")

            # Best venue rank
            if papers:
                ranks = [p.get("venue_rank", "?") for p in papers]
            else:
                ranks = ["?"] * len(raw_pubs)

            rank_priority = {"A*": 0, "A": 1, "B": 2, "C": 3, "?": 4}
            best_rank = min(ranks, key=lambda x: rank_priority.get(x, 4)) if ranks else "?"
            venue_text = Text(RANK_BADGES.get(best_rank, best_rank), style=RANK_COLORS.get(best_rank, "white"))

            # Education
            if r.education:
                top_edu = r.education[0]
                edu_str = f"{top_edu.degree}\n{top_edu.institution[:16]}"
            else:
                edu_str = "—"
            edu_text = Text(edu_str, style="dim white")

            # Score breakdown mini
            s = r.scores
            score_mini = (
                f"[cyan]{s.get('publications', 0):.0f}[/cyan]"
                f"·[blue]{s.get('research', 0):.0f}[/blue]"
                f"·[green]{s.get('education', 0):.0f}[/green]"
                f"·[yellow]{s.get('fit', 0):.0f}[/yellow]"
                f"·[magenta]{s.get('trajectory', 0):.0f}[/magenta]"
            )

            if r.llm_error:
                table.add_row(rank_str, Text(r.name, style="dim red"), "ERR", "—", "—", "—", "—", "—")
            else:
                table.add_row(
                    rank_str, name, total_text, bar,
                    papers_text, venue_text, edu_text, score_mini,
                )

        self.console.print(table)
        self.console.print(
            "[dim]Score legend: [cyan]Pubs[/cyan] · [blue]Research[/blue] · "
            "[green]Education[/green] · [yellow]Fit[/yellow] · [magenta]Trajectory[/magenta]  "
            "| Venue: [bold magenta]A*[/bold magenta] > [bold cyan]A[/bold cyan] > "
            "[bold yellow]B[/bold yellow] > [dim yellow]C[/dim yellow][/dim]"
        )

    # ── Candidate Profile ──────────────────────────────────────────────────────

    def _render_candidate_profile(self, r):
        medal = MEDAL.get(r.rank, f"#{r.rank}")
        header = f"{medal}  {r.name}"
        if r.email:
            header += f"  ·  {r.email}"
        if r.homepage:
            header += f"  ·  {r.homepage}"

        self.console.print(Rule(f"[bold cyan]{header}[/bold cyan]", style="cyan"))

        # Score breakdown
        self._render_score_breakdown(r)

        # Research summary
        if r.research_summary:
            self.console.print(Panel(
                f"[white]{r.research_summary}[/white]",
                title="[bold]Research Summary[/bold]",
                border_style="blue",
                padding=(0, 2),
            ))

        # Education
        if r.education:
            self._render_education(r)

        # Research experience
        if r.research_experience:
            self._render_research_experience(r)

        # Publications + timeline
        if r.publications or r.enriched_papers:
            self._render_publications(r)

        # Skills & Awards
        row_items = []
        if r.skills:
            skills_text = ", ".join(r.skills[:15])
            row_items.append(Panel(f"[cyan]{skills_text}[/cyan]",
                                   title="[bold]Skills[/bold]", border_style="dim cyan",
                                   padding=(0, 1)))
        if r.awards:
            awards_text = "\n".join(f"🏅 {a}" for a in r.awards[:5])
            row_items.append(Panel(awards_text, title="[bold]Awards[/bold]",
                                   border_style="dim yellow", padding=(0, 1)))
        if row_items:
            self.console.print(Columns(row_items, equal=False, expand=True))

        # Fit notes
        if r.fit_notes:
            self.console.print(Panel(
                f"[italic yellow]{r.fit_notes}[/italic yellow]",
                title="[bold yellow]Role Fit Assessment[/bold yellow]",
                border_style="yellow", padding=(0, 2),
            ))

        # Strengths / Weaknesses
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
            score = r.scores.get(key, 0)
            bar = score_bar(score, width=24)
            lines.append(
                f"  [{color}]{label:<16}[/{color}] {bar} [{score_color(score)}]{score:5.1f}[/{score_color(score)}]"
            )

        total_bar = score_bar(r.total_score, width=24)
        lines.append(f"  [bold white]{'TOTAL':<16}[/bold white] {total_bar} [bold white]{r.total_score:5.1f}[/bold white]")

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Score Breakdown[/bold]",
            border_style="cyan",
            padding=(0, 1),
        ))

    def _render_education(self, r):
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                      border_style="dim", padding=(0, 2))
        table.add_column("Degree")
        table.add_column("Field")
        table.add_column("Institution")
        table.add_column("Year", justify="right")
        table.add_column("GPA", justify="right")

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
            tag = "[ACad]" if exp.is_academic else "[Indus]"
            header = Text()
            header.append(f"{tag} ", style=f"dim {color}")
            header.append(exp.role or "Researcher", style="bold white")
            header.append(f" @ {exp.organization}", style=f"bold {color}")
            header.append(f"  [{exp.duration}]", style="dim")
            if exp.description:
                header.append(f"\n   {exp.description[:120]}", style="dim white")
            items.append(header)

        panel_content = Text("\n\n").join(items) if items else Text("None")
        self.console.print(Panel(panel_content, title="[bold]Research Experience[/bold]",
                                 border_style="blue", padding=(0, 1)))

    def _render_publications(self, r):
        papers = r.enriched_papers or []
        raw_pubs = r.publications or []

        # Build timeline data
        year_map: dict[int, list] = {}

        if papers:
            for p in papers:
                yr = p.get("year") or 0
                year_map.setdefault(yr, []).append(p)
        else:
            for p in raw_pubs:
                yr = p.year or 0
                year_map.setdefault(yr, []).append({
                    "title": p.title,
                    "venue": p.venue,
                    "venue_rank": "?",
                    "citation_count": 0,
                    "is_first_author": p.is_first_author,
                })

        # Publications table
        table = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold dim", padding=(0, 1))
        table.add_column("Year", width=6, justify="center")
        table.add_column("Venue", width=8, justify="center")
        table.add_column("Title", min_width=40)
        table.add_column("★ First", width=7, justify="center")
        table.add_column("Cites", width=7, justify="right")
        table.add_column("SS Link", width=12)

        all_papers = sorted(
            (p for yr, ps in year_map.items() for p in ps),
            key=lambda x: x.get("year") or 0,
            reverse=True,
        )

        for p in all_papers:
            rank = p.get("venue_rank", "?")
            rank_style = RANK_COLORS.get(rank, "white")
            rank_label = RANK_BADGES.get(rank, rank)

            title = p.get("title", "Unknown")[:70]
            cites = p.get("citation_count", 0) or 0
            first = "✓" if p.get("is_first_author") else ""
            ss_url = p.get("semantic_scholar_url")
            ss_link = Text("[SS]", style="link " + ss_url) if ss_url else Text("—", style="dim")

            table.add_row(
                str(p.get("year") or "?"),
                Text(rank_label, style=rank_style),
                Text(title, style="white"),
                Text(first, style="bold green"),
                Text(str(cites) if cites else "—", style="cyan"),
                ss_link,
            )

        # Timeline visualization
        timeline = self._build_timeline(year_map)

        self.console.print(Panel(
            table,
            title=f"[bold]Publications ({len(all_papers)})[/bold]",
            border_style="magenta",
            padding=(0, 0),
        ))

        if timeline:
            self.console.print(Panel(
                timeline,
                title="[bold]Publication Timeline[/bold]",
                border_style="dim magenta",
                padding=(0, 1),
            ))

    def _build_timeline(self, year_map: dict) -> Optional[Text]:
        valid_years = sorted(yr for yr in year_map if yr > 1990)
        if not valid_years:
            return None

        min_y, max_y = min(valid_years), max(valid_years)
        if min_y == max_y:
            valid_years = [min_y]

        lines = []
        rank_priority = {"A*": 0, "A": 1, "B": 2, "C": 3, "?": 4}

        for yr in range(min_y, max_y + 1):
            papers = year_map.get(yr, [])
            count = len(papers)
            if count == 0:
                bar_segment = Text(f"  {yr}  ", style="dim")
                bar_segment.append("·", style="dim white")
                lines.append(bar_segment)
                continue

            best_rank = min(
                (p.get("venue_rank", "?") for p in papers),
                key=lambda x: rank_priority.get(x, 4)
            )
            color = {"A*": "magenta", "A": "cyan", "B": "yellow", "C": "white", "?": "dim"}[best_rank]

            bar = "█" * min(count * 3, 30)
            line = Text(f"  {yr}  ", style="dim white")
            line.append(bar, style=color)
            line.append(f"  {count} paper{'s' if count != 1 else ''}", style="dim")
            line.append(f"  [best: {best_rank}]", style=f"dim {color}")
            lines.append(line)

        result = Text("\n").join(lines)
        return result

    def _render_swot(self, r):
        items = []
        if r.strengths:
            s_text = "\n".join(f"[green]▲[/green] {s}" for s in r.strengths)
            items.append(Panel(s_text, title="[bold green]Strengths[/bold green]",
                               border_style="green", padding=(0, 1)))
        if r.weaknesses:
            w_text = "\n".join(f"[red]▼[/red] {w}" for w in r.weaknesses)
            items.append(Panel(w_text, title="[bold red]Weaknesses[/bold red]",
                               border_style="red", padding=(0, 1)))
        if items:
            self.console.print(Columns(items, equal=True, expand=True))

    def _render_error_card(self, r):
        self.console.print(Panel(
            f"[red]Error:[/red] {r.llm_error}",
            title=f"[red]✗ {r.name}[/red]",
            border_style="red",
        ))

    # ── Summary Stats ─────────────────────────────────────────────────────────

    def _render_summary_stats(self, ranked: list):
        valid = [r for r in ranked if not r.llm_error]
        if not valid:
            return

        self.console.print(Rule("[bold]Summary Statistics[/bold]", style="dim"))

        total_papers = sum(len(r.publications) for r in valid)
        astar_papers = sum(
            1 for r in valid
            for p in r.enriched_papers
            if p.get("venue_rank") == "A*"
        )
        avg_score = sum(r.total_score for r in valid) / len(valid)

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold white")

        table.add_row("Candidates processed", str(len(valid)))
        table.add_row("Total publications", str(total_papers))
        table.add_row("A* venue papers", f"[magenta]{astar_papers}[/magenta]")
        table.add_row("Average total score", f"{avg_score:.1f}")
        table.add_row("Top candidate", f"[bold cyan]{valid[0].name}[/bold cyan] ({valid[0].total_score:.1f})" if valid else "—")

        self.console.print(Padding(table, (1, 4)))

    # ── API Usage Stats ──────────────────────────────────────────────────────

    def render_api_usage(self, usage: dict):
        """Display Groq API usage / rate limit summary."""
        self.console.print(Rule("[bold]Groq API Usage[/bold]", style="dim"))

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold white")

        table.add_row("Model", f"[cyan]{usage.get('model', '?')}[/cyan]")
        table.add_row("API calls made", str(usage.get("requests_made", 0)))
        table.add_row(
            "Requests remaining (daily)",
            f"{usage.get('requests_remaining', '?'):,} / {usage.get('requests_limit_daily', '?'):,}",
        )

        tokens_used = usage.get("tokens_used", 0)
        tokens_daily = usage.get("tokens_limit_daily", 1)
        pct = (tokens_used / tokens_daily * 100) if tokens_daily else 0
        table.add_row(
            "Tokens used",
            f"{tokens_used:,}  [dim]({pct:.1f}% of daily {tokens_daily:,})[/dim]",
        )
        table.add_row("  Prompt tokens", f"{usage.get('prompt_tokens', 0):,}")
        table.add_row("  Completion tokens", f"{usage.get('completion_tokens', 0):,}")
        table.add_row("Avg tokens/call", f"{usage.get('avg_tokens_per_call', 0):,}")
        table.add_row(
            "Rate limit waits",
            f"{usage.get('rate_limit_waits', 0)}  "
            f"[dim]({usage.get('wait_seconds', 0):.1f}s total)[/dim]",
        )
        table.add_row("Retries (429/503)", str(usage.get("retries", 0)))
        table.add_row(
            "Limits",
            f"[dim]RPM={usage.get('rpm', '?')}  TPM={usage.get('tpm', '?'):,}[/dim]",
        )

        self.console.print(Padding(table, (1, 4)))

    # ── JSON Export ───────────────────────────────────────────────────────────

    def export_json(self, ranked: list, path: str):
        output = []
        for r in ranked:
            record = {
                "rank": r.rank,
                "name": r.name,
                "email": r.email,
                "homepage": r.homepage,
                "total_score": r.total_score,
                "scores": r.scores,
                "research_summary": r.research_summary,
                "fit_notes": r.fit_notes,
                "strengths": r.strengths,
                "weaknesses": r.weaknesses,
                "education": [e.__dict__ for e in r.education],
                "research_experience": [e.__dict__ for e in r.research_experience],
                "publications_raw": [p.__dict__ for p in r.publications],
                "publications_enriched": r.enriched_papers,
                "skills": r.skills,
                "awards": r.awards,
                "source_file": str(r.raw.path),
                "error": r.llm_error,
            }
            output.append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
