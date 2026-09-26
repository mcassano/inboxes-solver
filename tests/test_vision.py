import json

import pytest

from inboxes import openrouter, vision


def test_image_to_data_url_png(tmp_path):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    data_url = vision._image_to_data_url(img_path)
    assert data_url.startswith("data:image/png;base64,")


def test_decode_screenshot_parses_valid_response(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    puzzle_json = {
        "rows": 2,
        "cols": 2,
        "clues": [
            {"row": 0, "col": 0, "value": 2},
            {"row": 0, "col": 1, "value": 2},
        ],
    }

    def fake_chat(model, messages, timeout=120, temperature=0):
        assert model == vision.DEFAULT_MODEL
        # Sanity check the image was embedded as a data URL in the payload.
        image_content = messages[1]["content"][1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"].startswith("data:image/png;base64,")
        return f"```json\n{json.dumps(puzzle_json)}\n```"

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path)
    assert puzzle.rows == 2
    assert puzzle.cols == 2
    assert len(puzzle.clues) == 2


def test_decode_screenshot_raises_on_invalid_puzzle(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    img_path.write_bytes(b"fake-png-bytes")

    # Area sum (3) does not match the grid size (4), so validate() must fail.
    bad_json = {"rows": 2, "cols": 2, "clues": [{"row": 0, "col": 0, "value": 3}]}
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: json.dumps(bad_json))

    with pytest.raises(ValueError, match="clue areas sum"):
        vision.decode_screenshot(img_path)
