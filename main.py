#!/usr/bin/env python3
"""
ResearchRank - Mass-ranking research resumes/CVs for academic labs.
Usage: python main.py --resumes ./cvs/ --job "ML researcher focusing on NLP"
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from parsers.resume_parser import ResumeParser
from analyzers.llm_analyzer import LLMAnalyzer
from analyzers.paper_analyzer import PaperAnalyzer
from rankers.scorer import Scorer
from reporters.terminal_reporter import TerminalReporter
from utils.rate_limiter import GroqRateLimiter
from config import Config

console = Console()


def print_banner():
    banner = Text()
    banner.append("  ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗\n", style="bold cyan")
    banner.append("  ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║\n", style="bold cyan")
    banner.append("  ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║\n", style="bold blue")
    banner.append("  ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║\n", style="bold blue")
    banner.append("  ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║\n", style="bold magenta")
    banner.append("  ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝\n", style="bold magenta")
    banner.append("              R A N K  ·  A C A D E M I C  ·  T A L E N T\n", style="dim white")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResearchRank — Academic CV ranking for labs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --resumes ./cvs/
  python main.py --resumes ./cvs/ --job "NLP researcher with LLM experience"
  python main.py --resumes ./cvs/ --top 5 --weights research=0.4,publications=0.4,education=0.2
  python main.py --resumes ./cvs/ --export results.json
        """
    )
    parser.add_argument("--resumes", "-r", required=True,
                        help="Path to folder containing CVs, or a single file")
    parser.add_argument("--job", "-j", default=None,
                        help="Job/role description to match candidates against")
    parser.add_argument("--top", "-t", type=int, default=None,
                        help="Show only top N candidates")
    parser.add_argument("--weights", "-w", default=None,
                        help="Custom scoring weights e.g. research=0.4,publications=0.3,education=0.2,fit=0.1")
    parser.add_argument("--export", "-e", default=None,
                        help="Export full results to JSON file")
    parser.add_argument("--no-papers", action="store_true",
                        help="Skip paper lookup (faster, no API calls to Semantic Scholar)")
    parser.add_argument("--model", "-m", default=None,
                        help="Override Groq model (free-tier options: "
                             "meta-llama/llama-4-scout-17b-16e-instruct [default, 30K TPM], "
                             "llama-3.1-8b-instant [14.4K RPD], "
                             "llama-3.3-70b-versatile [best quality, tight limits])")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed processing logs")
    return parser.parse_args()


def parse_weights(weights_str: str) -> dict:
    weights = {}
    for part in weights_str.split(","):
        k, v = part.strip().split("=")
        weights[k.strip()] = float(v.strip())
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        console.print(f"[yellow]⚠ Weights sum to {total:.2f}, normalizing to 1.0[/yellow]")
        weights = {k: v / total for k, v in weights.items()}
    return weights


def collect_resume_files(path_str: str) -> list[Path]:
    path = Path(path_str)
    supported = {".pdf", ".docx", ".doc", ".txt", ".md"}
    if path.is_file():
        return [path] if path.suffix.lower() in supported else []
    elif path.is_dir():
        files = []
        for ext in supported:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"**/*{ext}"))
        return sorted(set(files))
    else:
        console.print(f"[red]✗ Path not found: {path_str}[/red]")
        sys.exit(1)


def main():
    print_banner()
    args = parse_args()
    config = Config()

    if args.model:
        config.groq_model = args.model

    weights = None
    if args.weights:
        try:
            weights = parse_weights(args.weights)
        except Exception as e:
            console.print(f"[red]✗ Invalid weights format: {e}[/red]")
            sys.exit(1)

    # Collect files
    resume_files = collect_resume_files(args.resumes)
    if not resume_files:
        console.print("[red]✗ No supported resume files found (PDF, DOCX, TXT, MD)[/red]")
        sys.exit(1)

    console.print(f"\n[bold green]✓[/bold green] Found [bold]{len(resume_files)}[/bold] resume(s) to process\n")

    # Parse resumes
    parser = ResumeParser(verbose=args.verbose)
    parsed_resumes = parser.parse_all(resume_files)

    # Create rate limiter for Groq API
    rate_limiter = GroqRateLimiter(
        model=config.groq_model,
        verbose=args.verbose,
    )

    # LLM Analysis
    analyzer = LLMAnalyzer(config, verbose=args.verbose, rate_limiter=rate_limiter)
    analyzed = analyzer.analyze_all(parsed_resumes, job_description=args.job)

    # Paper lookup & venue ranking
    if not args.no_papers:
        paper_analyzer = PaperAnalyzer(verbose=args.verbose)
        analyzed = paper_analyzer.enrich_all(analyzed)

    # Score & rank
    scorer = Scorer(weights=weights, job_description=args.job)
    ranked = scorer.rank_all(analyzed)

    # Report
    reporter = TerminalReporter(console)
    reporter.render(ranked, top_n=args.top, job_description=args.job)

    # API Usage summary
    reporter.render_api_usage(rate_limiter.get_summary())

    # Export
    if args.export:
        reporter.export_json(ranked, args.export)
        console.print(f"\n[bold green]✓[/bold green] Results exported to [cyan]{args.export}[/cyan]")


if __name__ == "__main__":
    main()
