import base64
import io
import json

import cv2
import pytest
from PIL import Image

from inboxes import openrouter, vision


def _write_test_image(path, size=(40, 20), color=(255, 255, 255)):
    Image.new("RGB", size, color).save(path)


def test_prepare_image_returns_data_url(tmp_path):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    data_url = vision._prepare_image(img_path)
    assert data_url.startswith("data:image/png;base64,")


def test_prepare_image_crops_whitespace_and_upscales(tmp_path):
    img_path = tmp_path / "puzzle.png"
    # A 10x10 black square sitting in a much larger white canvas.
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    for x in range(80, 90):
        for y in range(80, 90):
            img.putpixel((x, y), (0, 0, 0))
    img.save(img_path)

    data_url = vision._prepare_image(img_path)
    header, b64 = data_url.split(",", 1)
    assert header == "data:image/png;base64"
    decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
    # Cropped to roughly the 10x10 square (plus padding) then upscaled 2x,
    # so it should end up much smaller than the original 200x200 canvas.
    assert decoded.width < 100
    assert decoded.height < 100


def test_prepare_image_handles_rgba_input(tmp_path):
    img_path = tmp_path / "puzzle.png"
    Image.new("RGBA", (30, 30), (10, 10, 10, 255)).save(img_path)
    data_url = vision._prepare_image(img_path)
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


# A 2x2 grid with clues (0,0)=2 and (0,1)=2: each clue's only geometrically
# possible rectangle (that doesn't swallow the other clue) is a vertical
# 2x1 strip in its own column, so this has exactly one solution.
UNIQUE_GRID_JSON = {"grid": [[2, 2], [None, None]]}

# A 2x3 grid with clues (0,0)=3 and (0,2)=3: the only geometrically possible
# rectangle for either clue is the entire top row, which would swallow the
# other clue -- so neither clue has any valid placement at all.
UNSOLVABLE_GRID_JSON = {"grid": [[3, None, 3], [None, None, None]]}

# A single clue covering the whole 2x3 grid: trivially, uniquely solvable.
SINGLE_CLUE_GRID_JSON = {"grid": [[6, None, None], [None, None, None]]}


def test_decode_screenshot_parses_valid_response(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)

    def fake_chat(model, messages, **kwargs):
        assert model == vision.DEFAULT_MODEL
        # Sanity check the image was embedded as a data URL in the payload.
        image_content = messages[1]["content"][1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"].startswith("data:image/png;base64,")
        return f"```json\n{json.dumps(UNIQUE_GRID_JSON)}\n```"

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path)
    assert puzzle.rows == 2
    assert puzzle.cols == 2
    assert len(puzzle.clues) == 2


def test_decode_screenshot_retries_same_model_before_falling_back(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", ["model-b"])

    # Area sum (3) does not match the grid size (2x2=4), so validate() fails
    # on the first attempt; the second attempt (same model) succeeds.
    bad_json = {"grid": [[3, None], [None, None]]}
    responses = [json.dumps(bad_json), json.dumps(UNIQUE_GRID_JSON)]
    calls = []

    def fake_chat(model, messages, **kwargs):
        calls.append(model)
        return responses.pop(0)

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path, model="model-a", attempts_per_model=2)
    assert len(puzzle.clues) == 2
    assert calls == ["model-a", "model-a"]  # never needed to fall back


def test_decode_screenshot_falls_back_to_a_different_model(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", ["model-b"])

    bad_json = {"grid": [[3, None], [None, None]]}

    def fake_chat(model, messages, **kwargs):
        # model-a keeps making the same mistake every time; model-b gets it
        # right immediately -- mirrors a model with a consistent habit of
        # misreading a particular puzzle, where switching models (not
        # re-sampling the same one) is what actually fixes it.
        if model == "model-a":
            return json.dumps(bad_json)
        return json.dumps(UNIQUE_GRID_JSON)

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    puzzle = vision.decode_screenshot(img_path, model="model-a", attempts_per_model=2)
    assert len(puzzle.clues) == 2


def test_decode_screenshot_exhausts_every_model_then_raises(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", ["model-b"])

    bad_json = {"grid": [[3, None], [None, None]]}
    calls = []

    def fake_chat(model, messages, **kwargs):
        calls.append(model)
        return json.dumps(bad_json)

    monkeypatch.setattr(openrouter, "chat", fake_chat)

    with pytest.raises(vision.VisionDecodeError, match="model-a, model-b"):
        vision.decode_screenshot(img_path, model="model-a", attempts_per_model=2)
    assert calls == ["model-a", "model-a", "model-b", "model-b"]


def test_decode_screenshot_rejects_structurally_valid_but_unsolvable_puzzle(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", [])

    # Sum (3 + 3 = 6) matches the grid size (2x3 = 6), so validate() passes,
    # but this is the "clue shifted to the wrong cell" failure mode that a
    # sum check alone can't catch -- only the solver oracle catches it.
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: json.dumps(UNSOLVABLE_GRID_JSON))

    with pytest.raises(vision.VisionDecodeError, match="model-a"):
        vision.decode_screenshot(img_path, model="model-a", attempts_per_model=1)


def test_decode_screenshot_recovers_after_unsolvable_attempt(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", [])

    responses = [json.dumps(UNSOLVABLE_GRID_JSON), json.dumps(SINGLE_CLUE_GRID_JSON)]
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: responses.pop(0))

    puzzle = vision.decode_screenshot(img_path, model="model-a", attempts_per_model=2)
    assert len(puzzle.clues) == 1
    assert responses == []


def test_decode_screenshot_rejects_ambiguous_puzzle(tmp_path, monkeypatch):
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", [])

    # A 2x2 grid with clues (0,0)=2 and (1,1)=2 diagonally opposite: it can
    # be tiled either as two horizontal strips or two vertical strips, both
    # valid -- two distinct solutions, so it's genuinely ambiguous.
    ambiguous_json = {"grid": [[2, None], [None, 2]]}
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: json.dumps(ambiguous_json))

    with pytest.raises(vision.VisionDecodeError, match="model-a"):
        vision.decode_screenshot(img_path, model="model-a", attempts_per_model=1)


def test_decode_screenshot_recovers_from_truncated_response(tmp_path, monkeypatch):
    # A response that got cut off mid-generation (e.g. the model ran away
    # hallucinating extra rows and hit the token limit) has no closing
    # brace at all -- openrouter.extract_json raises OpenRouterError for
    # that, which is a different exception family than the ValueError family
    # everything else here raises, and it used to slip past the retry loop
    # and crash the whole CLI instead of triggering a retry/fallback.
    img_path = tmp_path / "puzzle.png"
    _write_test_image(img_path)
    monkeypatch.setattr(vision, "FALLBACK_MODELS", [])

    truncated = '{"grid": [[2, 2], [null, null'  # no closing brace
    responses = [truncated, json.dumps(UNIQUE_GRID_JSON)]
    monkeypatch.setattr(openrouter, "chat", lambda *a, **k: responses.pop(0))

    puzzle = vision.decode_screenshot(img_path, model="model-a", attempts_per_model=2)
    assert len(puzzle.clues) == 2
    assert responses == []


def test_prepare_for_decode_measures_a_real_grid(tmp_path):
    # A clean synthetic grid should be found, rectified and measured, so the
    # model can be told the dimensions rather than counting lines itself.
    import numpy as np
    from tests.test_grid_detect import synthetic_grid

    img_path = tmp_path / "grid.png"
    cv2.imwrite(str(img_path), synthetic_grid(rows=11, cols=9))
    data_url, dims = vision._prepare_for_decode(img_path)
    assert data_url.startswith("data:image/png;base64,")
    assert dims == (11, 9)


def test_prepare_for_decode_falls_back_when_no_grid_found(tmp_path):
    img_path = tmp_path / "plain.png"
    _write_test_image(img_path, size=(60, 40))
    data_url, dims = vision._prepare_for_decode(img_path)
    assert data_url.startswith("data:image/png;base64,")
    assert dims is None


def test_decode_once_states_measured_dimensions_in_the_prompt(monkeypatch):
    seen = {}

    def fake_chat(model, messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return json.dumps(UNIQUE_GRID_JSON)

    monkeypatch.setattr(openrouter, "chat", fake_chat)
    vision._decode_once("model-a", "data:image/png;base64,xx", dimensions=(2, 2))

    assert "EXACTLY 2 rows" in seen["prompt"]
    assert "EXACTLY 2 columns" in seen["prompt"]


def test_decode_once_without_dimensions_uses_the_counting_prompt(monkeypatch):
    seen = {}

    def fake_chat(model, messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return json.dumps(UNIQUE_GRID_JSON)

    monkeypatch.setattr(openrouter, "chat", fake_chat)
    vision._decode_once("model-a", "data:image/png;base64,xx", dimensions=None)

    assert seen["prompt"] == vision.SYSTEM_PROMPT
    # The sum invariant is the model's own check when nothing measured it.
    assert "equals (rows) x (columns)" in seen["prompt"]
