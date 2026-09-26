import json

from inboxes import vision
from inboxes.cli import main


def write_puzzle(path):
    path.write_text(
        json.dumps(
            {
                "rows": 4,
                "cols": 4,
                "clues": [
                    {"row": 0, "col": 0, "value": 4},
                    {"row": 0, "col": 2, "value": 4},
                    {"row": 2, "col": 0, "value": 4},
                    {"row": 2, "col": 2, "value": 4},
                ],
            }
        )
    )


def test_cli_solves_puzzle_file_end_to_end(tmp_path):
    puzzle_path = tmp_path / "puzzle.json"
    write_puzzle(puzzle_path)
    out_json = tmp_path / "solution.json"
    out_png = tmp_path / "solution.png"

    exit_code = main(
        [
            "--puzzle",
            str(puzzle_path),
            "--out",
            str(out_json),
            "--png",
            str(out_png),
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_png.exists()
    solved = json.loads(out_json.read_text())
    for clue in solved["clues"]:
        assert clue["r1"] - clue["r0"] + 1 in (1, 2, 4)


def test_cli_rejects_both_sources(tmp_path):
    puzzle_path = tmp_path / "puzzle.json"
    write_puzzle(puzzle_path)
    try:
        main(["--puzzle", str(puzzle_path), "--screenshot", str(puzzle_path)])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected argparse to reject mutually exclusive arguments")


def test_cli_requires_a_source():
    try:
        main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected argparse to require --puzzle or --screenshot")


def test_cli_reports_vision_decode_failure_cleanly(tmp_path, monkeypatch, capsys):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    def fake_decode(*args, **kwargs):
        raise vision.VisionDecodeError("could not decode a valid puzzle: boom")

    monkeypatch.setattr("inboxes.cli.decode_screenshot", fake_decode)

    exit_code = main(["--screenshot", str(img_path)])

    assert exit_code == 1
    assert "could not decode a valid puzzle" in capsys.readouterr().err
