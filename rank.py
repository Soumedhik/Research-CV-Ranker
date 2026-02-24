#!/usr/bin/env python3
"""
rank.py — Convenience entry point for ResearchRank.
Usage: python rank.py --resumes ./cvs/ --job "ML researcher focusing on NLP"

This simply delegates to main.main(). All arguments are passed through.
"""
from main import main

if __name__ == "__main__":
    main()
