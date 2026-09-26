"""Command-line entry point: screenshot or JSON in, solved JSON + PNG out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .capture import (
        DEFAULT_CAMERA_INDEX,
        DEFAULT_MOTION_THRESHOLD,
        DEFAULT_OUT_PATH as DEFAULT_CAPTURE_OUT,
        DEFAULT_SETTLE_FRAMES,
        DEFAULT_SHARPNESS_THRESHOLD,
        CaptureError,
        capture_from_webcam,
    )
except ImportError:
    # opencv-python isn't installed. --webcam just won't be usable (main()
    # reports that clearly below); --screenshot and --puzzle don't need it.
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_MOTION_THRESHOLD = 3.0
    DEFAULT_SETTLE_FRAMES = 10
    DEFAULT_SHARPNESS_THRESHOLD = 150.0
    DEFAULT_CAPTURE_OUT = Path("webcam_capture.png")
    CaptureError = RuntimeError
    capture_from_webcam = None

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
    source.add_argument(
        "--webcam",
        action="store_true",
        help="capture the puzzle from a webcam instead of a file (hold it up and hold still)",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"OpenRouter vision model used to decode a screenshot (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"which camera to use with --webcam (default: {DEFAULT_CAMERA_INDEX}); if it opens the "
        "wrong camera, try 1, 2, ...",
    )
    parser.add_argument(
        "--capture-out",
        type=Path,
        default=DEFAULT_CAPTURE_OUT,
        help=f"where to save the raw --webcam photo, for inspecting what was captured "
        f"(default: {DEFAULT_CAPTURE_OUT})",
    )
    parser.add_argument(
        "--sharpness-threshold",
        type=float,
        default=DEFAULT_SHARPNESS_THRESHOLD,
        help=f"--webcam: minimum focus score to accept a frame (default: {DEFAULT_SHARPNESS_THRESHOLD})",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=DEFAULT_MOTION_THRESHOLD,
        help=f"--webcam: maximum frame-to-frame movement to accept (default: {DEFAULT_MOTION_THRESHOLD})",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=DEFAULT_SETTLE_FRAMES,
        help=f"--webcam: consecutive good frames required before capturing "
        f"(default: {DEFAULT_SETTLE_FRAMES})",
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

    image_path = args.screenshot
    if args.webcam:
        if capture_from_webcam is None:
            print(
                "error: --webcam requires opencv-python (pip install -r requirements.txt)",
                file=sys.stderr,
            )
            return 1
        print("Waiting for a sharp, steady shot from the webcam (hold the puzzle up)...", file=sys.stderr)
        try:
            image_path = capture_from_webcam(
                camera_index=args.camera_index,
                out_path=args.capture_out,
                sharpness_threshold=args.sharpness_threshold,
                motion_threshold=args.motion_threshold,
                settle_frames=args.settle_frames,
            )
        except CaptureError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Captured {image_path}", file=sys.stderr)

    if image_path:
        print(f"Decoding {image_path} with {args.vision_model}...", file=sys.stderr)
        try:
            puzzle = decode_screenshot(image_path, model=args.vision_model)
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
