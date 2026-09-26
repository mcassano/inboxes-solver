import json

import pytest

from inboxes import openrouter, vision


def test_image_to_data_url_png(tmp_path):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    data_url = vision._image_to_data_url(img_path)
    assert data_url.startswith("data:image/png;base64,")


def test_grid_to_puzzle_converts_dense_grid():
    data = {"grid": [[None, 2], [2, None]]}
    puzzle = vision._grid_to_puzzle(data)
    assert puzzle.rows == 2
    assert puzzle.cols == 2
    assert sorted((c.row, c.col, c.value) for c in puzzle.clues) == [(0, 1, 2), (1, 0, 2)]


def test_grid_to_puzzle_keeps_blank_rows():
    # A fully-blank middle row must still count towards "rows".
    data = {"grid": [[1], [None], [None] * 1]}
    puzzle = vision._grid_to_puzzle(data)
    assert puzzle.rows == 3


def test_grid_to_puzzle_rejects_ragged_rows():
    data = {"grid": [[None, None], [None]]}
    with pytest.raises(ValueError, match="row 1 has"):
        vision._grid_to_puzzle(data)


def test_grid_to_puzzle_rejects_missing_grid_key():
    with pytest.raises(KeyError):
        vision._grid_to_puzzle({})


def test_decode_screenshot_parses_valid_response(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    grid_json = {"grid": [[2, 2], [None, None]]}

    def fake_chat(model, messages, timeout=120, temperature=0):
        assert model == vision.DEFAULT_MODEL
        # Sanity check the image was embedded as a data URL in the payload.
        image_content = messages[1]["content"][1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"].startswith("data:image/png;base64,")
        return f"```json\n{json.dumps(grid_json)}\n```"

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path)
    assert puzzle.rows == 2
    assert puzzle.cols == 2
    assert len(puzzle.clues) == 2


def test_decode_screenshot_retries_then_raises_on_persistently_invalid_puzzle(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    # Area sum (3) does not match the grid size (2x2=4), so validate() must
    # fail every time -- this exercises the retry-with-feedback loop giving up.
    bad_json = {"grid": [[3, None], [None, None]]}
    calls = []

    def fake_chat(model, messages, timeout=120, temperature=0):
        calls.append(messages)
        return json.dumps(bad_json)

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    with pytest.raises(vision.VisionDecodeError, match="after 3 attempts"):
        vision.decode_screenshot(img_path, max_attempts=3)
    assert len(calls) == 3
    # Each retry should feed the previous bad answer and the error back in.
    assert calls[-1][-1]["role"] == "user"
    assert "invalid" in calls[-1][-1]["content"]


def test_decode_screenshot_recovers_after_one_bad_attempt(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    bad_json = {"grid": [[3, None], [None, None]]}
    good_json = {"grid": [[2, 2], [None, None]]}
    responses = [json.dumps(bad_json), json.dumps(good_json)]

    def fake_chat(model, messages, timeout=120, temperature=0):
        return responses.pop(0)

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path, max_attempts=3)
    assert len(puzzle.clues) == 2
    assert responses == []
