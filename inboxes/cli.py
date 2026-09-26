"""Command-line entry point: screenshot or JSON in, solved JSON + PNG out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm_solver import DEFAULT_MODEL as DEFAULT_LLM_SOLVER_MODEL
from .llm_solver import llm_solve
from .puzzle import Puzzle
from .render import render_puzzle
from .solver import Unsolvable, solve_puzzle, verify_solution
from .vision import DEFAULT_MODEL as DEFAULT_VISION_MODEL
from .vision import VisionDecodeError, decode_screenshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inboxes",
        description="Decode, solve, and render The Economist's Inboxes (Shikaku) puzzle.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--screenshot", type=Path, help="path to a screenshot of the puzzle")
    source.add_argument(
        "--puzzle", type=Path, help="path to a puzzle already in the standardized JSON format"
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"OpenRouter vision model used to decode a screenshot (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument(
        "--solver",
        choices=("rules", "llm"),
        default="rules",
        help="solving method: 'rules' (deterministic, default) or 'llm' (ask a model on OpenRouter)",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_SOLVER_MODEL,
        help=f"OpenRouter model used when --solver llm is selected (default: {DEFAULT_LLM_SOLVER_MODEL})",
    )
    parser.add_argument(
        "--decoded-out",
        type=Path,
        help="where to save the decoded (unsolved) puzzle JSON, if decoding a screenshot",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("solution.json"), help="where to save the solved puzzle JSON"
    )
    parser.add_argument(
        "--png", type=Path, default=Path("solution.png"), help="where to save the solution image"
    )
    parser.add_argument(
        "--puzzle-png", type=Path, help="optional: also render the unsolved puzzle to this PNG"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.screenshot:
        print(f"Decoding {args.screenshot} with {args.vision_model}...", file=sys.stderr)
        try:
            puzzle = decode_screenshot(args.screenshot, model=args.vision_model)
        except VisionDecodeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print(
                "hint: transcribe the puzzle by hand into the JSON format "
                "described in the README and pass it with --puzzle instead.",
                file=sys.stderr,
            )
            return 1
        if args.decoded_out:
            puzzle.save(args.decoded_out)
            print(f"Wrote decoded puzzle to {args.decoded_out}", file=sys.stderr)
    else:
        puzzle = Puzzle.load(args.puzzle)
        puzzle.validate()

    if args.puzzle_png:
        render_puzzle(puzzle, args.puzzle_png)
        print(f"Wrote puzzle image to {args.puzzle_png}", file=sys.stderr)

    if args.solver == "llm":
        print(f"Solving with LLM ({args.llm_model})...", file=sys.stderr)
        solved = llm_solve(puzzle, model=args.llm_model)
    else:
        print("Solving with deterministic solver...", file=sys.stderr)
        try:
            solved = solve_puzzle(puzzle)
        except Unsolvable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    verify_solution(solved)
    solved.save(args.out)
    render_puzzle(solved, args.png)
    print(f"Wrote solution to {args.out} and {args.png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
