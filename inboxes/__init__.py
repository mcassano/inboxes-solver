"""inboxes: decode, solve, and render The Economist's "Inboxes" puzzle."""

from .puzzle import Clue, Puzzle
from .solver import ShikakuSolver, Unsolvable, solve_puzzle, verify_solution

__all__ = [
    "Clue",
    "Puzzle",
    "ShikakuSolver",
    "Unsolvable",
    "solve_puzzle",
    "verify_solution",
]
