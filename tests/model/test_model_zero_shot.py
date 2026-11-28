"""Issue #47: the zero-shot Apertus predictor. A fake backend and synthetic images: no network, no competition data."""

from __future__ import annotations

import io
import json
import math
import threading

import pytest
from model_synthetic import DIMENSIONS, FAKE_CLASSES, FakeMetric, fake_data_lane, train_table
from PIL import Image

from galaxus.data.images import prepared_jpeg
from galaxus.data.label_errors import out_of_fold_frame
from galaxus.model import apertus
from galaxus.model import eval as model_eval
from galaxus.model import zero_shot as zero_shot_module
from galaxus.model.apertus.backend import BackendError, Generation
from galaxus.model.zero_shot import (
    IMAGE_FIRST,
    INSTRUCTION,
    MACRO_AWARE,
    NO_CONTEXT,
    OBSERVATION_FIELD,
    OBSERVE_FIRST,
    OBSERVE_MAX_TOKENS,
    PRESETS,
    ZeroShotPredictor,
    answer_schema,
    cropped_jpeg,
    parse_answer,
    system_prompt,
    value_confidence,
)
from galaxus.types import ImageItem

ALLOWED = {dimension: frozenset(names) for dimension, names in FAKE_CLASSES.items()}
DEFAULTS = {dimension: names[0] for dimension, names in FAKE_CLASSES.items()}
GOOD = {dimension: names[2] for dimension, names in FAKE_CLASSES.items()}


class FakeBackend:
    """Answers with the scripted texts in turn (the last one repeats) and records every request."""

    name = "fake"
    model = "apertus-test"

    def __init__(self, *answers: str | Generation | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, images=(), json_schema=None, *, max_tokens=256, temperature=0.0):
        with self._lock:
            self.calls.append(
                {"messages": messages, "images": list(images), "schema": json_schema, "t": temperature, "max": max_tokens}
            )
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, Generation):
            return answer
        return Generation(text=answer, prompt_tokens=100, completion_tokens=10)


def with_logprobs(answer: dict[str, str], logprob: float = -0.1) -> Generation:
    """``answer`` as JSON, tokenised the way the gateway does (probe on #75): every class name one token at
    ``logprob``, the JSON around it at 0, and an end-of-sequence token that is not in the text."""
    pieces = [("{", 0.0)]
    for index, (dimension, label) in enumerate(answer.items()):
        pieces += [((", " if index else "") + f'"{dimension}": "', 0.0), (label, logprob), ('"', 0.0)]
    pieces += [("}", 0.0)]
    text = "".join(token for token, _ in pieces)
    assert json.loads(text) == answer
    tokens = [{"token": token, "logprob": value, "top_logprobs": []} for token, value in [*pieces, ("</s>", 0.0)]]
    return Generation(text=text, prompt_tokens=100, completion_tokens=len(tokens), logprobs=tokens)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A synthetic ``$GALAXUS_DATA``: the image cache and the guideline live there."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "labeling_guideline.md").write_text("# Synthetic guideline\nRule one.\n", encoding="utf-8")
    monkeypatch.setenv("GALAXUS_DATA", str(root))
    return root


def image_file(folder, name="img_00000"):
    """A transparent PNG: the predictor must send it composited on white, as a JPEG."""
    path = folder / f"{name}.png"
    Image.new("RGBA", (300, 200), (255, 0, 0, 0)).save(path)
    return path


def predictor(backend, **options):
    return ZeroShotPredictor(
        backend, defaults=DEFAULTS, dimensions=DIMENSIONS, allowed=ALLOWED, guideline="Rule one.", **options
    )


def item(data_dir, name="img_00000", context=None):
    return ImageItem(image_id=name, path=image_file(data_dir, name), context=context)


def test_the_schema_has_one_enum_of_allowed_classes_per_dimension():
    schema = answer_schema(DIMENSIONS, ALLOWED)
    assert schema["required"] == list(DIMENSIONS)
    assert schema["additionalProperties"] is False
    for dimension in DIMENSIONS:
        assert set(schema["properties"][dimension]["enum"]) == ALLOWED[dimension]


def test_parse_is_lenient_about_wrapping_but_strict_about_classes():
    text = "```json\n" + json.dumps({**GOOD, "subject": "not_a_class"}) + "\n```"
    parsed = parse_answer(text, DIMENSIONS, ALLOWED)
    assert "subject" not in parsed
    assert parsed["background"] == GOOD["background"]
    assert parse_answer("no json here", DIMENSIONS, ALLOWED) == {}
    assert parse_answer("{broken", DIMENSIONS, ALLOWED) == {}


def test_one_call_per_image_with_guideline_context_schema_and_a_jpeg(data_dir):
    backend = FakeBackend(json.dumps(GOOD))
    [prediction] = predictor(backend).predict([item(data_dir, context={"product_title": "Mouse"})])
    assert prediction.labels == GOOD
    [call] = backend.calls
    system, user = call["messages"]
    assert "Rule one." in system["content"] and system["role"] == "system"
    assert "product_title: Mouse" in user["content"]
    assert call["schema"] == answer_schema(DIMENSIONS, ALLOWED)
    assert call["t"] == 0.0
    assert call["images"][0][:3] == b"\xff\xd8\xff"


def test_without_context_the_context_is_not_sent(data_dir):
    backend = FakeBackend(json.dumps(GOOD))
    predictor(backend, use_context=False).predict([item(data_dir, context={"product_title": "Mouse"})])
    assert backend.calls[0]["messages"][1]["content"] == NO_CONTEXT


def test_the_default_system_prompt_is_unchanged_so_the_earlier_rows_keep_their_prompt_hash():
    """EXP-0002 and EXP-0009 recorded the hash of this exact text; the image-first option must not move it."""
    expected = INSTRUCTION.format(dimensions=", ".join(DIMENSIONS)) + "\n\nRule one."
    assert system_prompt("Rule one.\n", DIMENSIONS) == expected
    assert IMAGE_FIRST not in expected


def test_image_first_tells_the_model_to_trust_the_image_and_still_sends_the_context(data_dir):
    """#106, EXP-0014: one more line in the system prompt; the context stays in the user message."""
    backend = FakeBackend(json.dumps(GOOD))
    image_first = predictor(backend, image_first=True)
    [prediction] = image_first.predict([item(data_dir, context={"product_title": "Mouse"})])
    assert prediction.labels == GOOD
    system, user = backend.calls[0]["messages"]
    assert IMAGE_FIRST in system["content"] and "Rule one." in system["content"]
    assert system["content"].index(IMAGE_FIRST) < system["content"].index("Rule one.")
    assert "product_title: Mouse" in user["content"]
    plain = predictor(FakeBackend(json.dumps(GOOD)))
    assert image_first.prompt_sha256() != plain.prompt_sha256()
    assert image_first.config()["image_first"] is True
    assert plain.config()["image_first"] is False


def test_the_runner_can_build_the_image_first_predictor_without_arguments(monkeypatch):
    built = {}
    monkeypatch.setattr(zero_shot_module, "ZeroShotPredictor", lambda **options: built.update(options) or "built")
    assert zero_shot_module.image_first() == "built"
    assert built == {"image_first": True}


# --- #106: the score sprint's prompt and image variants ---------------------------------------------------------


def test_the_default_schema_is_unchanged_and_observe_first_puts_the_observation_first():
    plain = answer_schema(DIMENSIONS, ALLOWED)
    assert list(plain["properties"]) == plain["required"] == list(DIMENSIONS)
    observed = answer_schema(DIMENSIONS, ALLOWED, observe_first=True)
    assert list(observed["properties"]) == observed["required"] == [OBSERVATION_FIELD, *DIMENSIONS]
    assert observed["additionalProperties"] is False
    assert observed["properties"][DIMENSIONS[0]] == plain["properties"][DIMENSIONS[0]]


def test_each_prompt_line_is_added_once_in_a_fixed_order_and_image_first_alone_is_what_exp_0014_sends():
    text = system_prompt("Rule one.", DIMENSIONS, image_first=True, macro_aware=True, observe_first=True)
    assert text.index(IMAGE_FIRST) < text.index(MACRO_AWARE) < text.index(OBSERVE_FIRST) < text.index("Rule one.")
    assert all(text.count(line) == 1 for line in (IMAGE_FIRST, MACRO_AWARE, OBSERVE_FIRST))
    instruction = INSTRUCTION.format(dimensions=", ".join(DIMENSIONS))
    assert system_prompt("Rule one.", DIMENSIONS, image_first=True) == f"{instruction}\n{IMAGE_FIRST}\n\nRule one."


def test_observe_first_asks_for_an_observation_with_room_and_still_reads_the_labels(data_dir):
    backend = FakeBackend(json.dumps({OBSERVATION_FIELD: "A black mouse on white, seen from above.", **GOOD}))
    observe = predictor(backend, observe_first=True)
    [prediction] = observe.predict([item(data_dir)])
    assert prediction.labels == GOOD
    [call] = backend.calls
    assert call["schema"] == answer_schema(DIMENSIONS, ALLOWED, observe_first=True)
    assert call["max"] == OBSERVE_MAX_TOKENS
    assert OBSERVE_FIRST in call["messages"][0]["content"]
    assert observe.config()["max_tokens"] == OBSERVE_MAX_TOKENS and observe.config()["defaulted_cells"] == 0


def test_no_title_drops_the_title_keeps_the_rest_and_falls_back_to_no_context(data_dir):
    backend = FakeBackend(json.dumps(GOOD))
    no_title = predictor(backend, drop_context=("product_title",))
    no_title.predict([item(data_dir, "img_a", context={"product_title": "Mouse", "brand": "Acme"})])
    no_title.predict([item(data_dir, "img_b", context={"product_title": "Mouse"})])
    first, second = (call["messages"][1]["content"] for call in backend.calls)
    assert "brand: Acme" in first and "Mouse" not in first
    assert second == NO_CONTEXT
    assert no_title.config()["drop_context"] == ["product_title"]


def _decoded(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _red_share(image: Image.Image) -> float:
    raw = image.tobytes()
    pixels = list(zip(raw[0::3], raw[1::3], raw[2::3], strict=True))
    return sum(1 for r, g, b in pixels if r > 180 and g < 90 and b < 90) / len(pixels)


def test_crop_cuts_the_white_margins_so_the_product_fills_the_image(tmp_path):
    path = tmp_path / "small_product.png"
    image = Image.new("RGB", (1000, 800), (255, 255, 255))
    image.paste((210, 30, 30), (400, 300, 600, 500))
    image.save(path)
    cropped, plain = _decoded(cropped_jpeg(path)), _decoded(prepared_jpeg(path))
    assert _red_share(plain) < 0.1
    assert _red_share(cropped) > 0.6
    assert abs(cropped.width - cropped.height) <= 32


@pytest.mark.parametrize(
    "picture",
    [
        Image.new("RGB", (600, 400), (30, 120, 200)),  # a coloured background fills the frame: nothing to crop
        Image.new("RGB", (600, 400), (255, 255, 255)),  # blank: nothing to crop towards
        Image.new("RGBA", (300, 200), (255, 0, 0, 0)),  # fully transparent: white once composited
    ],
)
def test_crop_leaves_an_image_without_white_margins_exactly_as_the_data_lane_prepares_it(tmp_path, picture):
    path = tmp_path / "picture.png"
    picture.save(path)
    assert cropped_jpeg(path) == prepared_jpeg(path)


def test_the_crop_variant_sends_the_cropped_image(data_dir, monkeypatch):
    monkeypatch.setattr(zero_shot_module, "cropped_jpeg", lambda path, image_size: b"\xff\xd8\xffcropped")
    backend = FakeBackend(json.dumps(GOOD))
    crop = predictor(backend, image="crop")
    crop.predict([item(data_dir)])
    assert backend.calls[0]["images"] == [b"\xff\xd8\xffcropped"]
    assert crop.config()["image"] == "crop"


def test_an_unknown_image_mode_is_refused():
    options = {"defaults": DEFAULTS, "dimensions": DIMENSIONS, "allowed": ALLOWED, "guideline": ""}
    with pytest.raises(ValueError, match="image must be one of"):
        ZeroShotPredictor(FakeBackend("{}"), image="sepia", **options)


@pytest.mark.parametrize(("name", "options"), sorted(PRESETS.items()))
def test_every_preset_runs_through_the_eval_script_and_records_its_options(tmp_path, data_dir, monkeypatch, name, options):
    monkeypatch.setattr(model_eval, "git_state", lambda: ("0" * 40, False))
    backend = FakeBackend(json.dumps(GOOD))
    monkeypatch.setattr(apertus, "backend", lambda *args, **kwargs: backend)
    train = train_table()
    train["image_file"] = [str(image_file(data_dir, image_id)) for image_id in train["image_id"]]
    data = fake_data_lane(train, tmp_path / "out", FakeMetric())
    if options.get("hints"):  # #237: the hints file this variant reads, one line per image
        hints = tmp_path / "hints.jsonl"
        hints.write_text("".join(json.dumps({"image_id": i, "hint": "Similar catalog images were labelled: x"}) + "\n"
                                 for i in train["image_id"]), encoding="utf-8")
        monkeypatch.setenv("ZERO_SHOT_HINTS", str(hints))
    row = model_eval.run_eval(name, data, results_path=tmp_path / "results" / "metrics.json", exp_id="EXP-0016")
    config = row["config"]
    assert row["predictor"] == name
    assert config["image"] == options.get("image", "plain")
    assert config["image_size"] == options.get("image_size", 512)
    assert config["drop_context"] == list(options.get("drop_context", ()))
    for flag in ("image_first", "macro_aware", "observe_first", "per_dimension"):
        assert config[flag] is options.get(flag, False)
    calls_per_image = len(DIMENSIONS) if options.get("per_dimension") else 1  # #156: one request per dimension
    assert config["images"] == row["holdout"]["images"] == len(backend.calls) // calls_per_image


def test_a_bad_answer_is_asked_once_more(data_dir):
    backend = FakeBackend("garbage", json.dumps(GOOD))
    zero_shot = predictor(backend)
    [prediction] = zero_shot.predict([item(data_dir)])
    assert prediction.labels == GOOD
    assert len(backend.calls) == 2
    assert zero_shot.config()["retries"] == 1
    assert zero_shot.config()["defaulted_cells"] == 0


def test_after_the_retry_only_the_bad_dimensions_get_defaults_and_are_counted(data_dir):
    partial = json.dumps({dimension: GOOD[dimension] for dimension in DIMENSIONS[:4]})
    backend = FakeBackend(partial, "garbage")
    zero_shot = predictor(backend)
    [prediction] = zero_shot.predict([item(data_dir)])
    assert len(backend.calls) == 2
    for dimension in DIMENSIONS[:4]:
        assert prediction.labels[dimension] == GOOD[dimension]
    for dimension in DIMENSIONS[4:]:
        assert prediction.labels[dimension] == DEFAULTS[dimension]
    assert zero_shot.config()["defaulted_cells"] == 2


def test_a_backend_error_is_raised_not_turned_into_defaults(data_dir):
    backend = FakeBackend(BackendError("api: HTTP 503", status=503))
    with pytest.raises(BackendError):
        predictor(backend).predict([item(data_dir)])


def test_parallel_requests_keep_the_order_of_the_batch(data_dir):
    items = [item(data_dir, f"img_{index:05d}") for index in range(12)]
    predictions = predictor(FakeBackend(json.dumps(GOOD)), workers=4).predict(items)
    assert [p.image_id for p in predictions] == [i.image_id for i in items]


def test_defaults_must_be_allowed_classes():
    with pytest.raises(ValueError, match="defaults"):
        ZeroShotPredictor(
            FakeBackend("{}"), defaults={**DEFAULTS, "subject": "x"}, dimensions=DIMENSIONS, allowed=ALLOWED, guideline=""
        )


@pytest.mark.parametrize(
    ("name", "context", "image_first"),
    [("zero-shot", True, False), ("zero-shot-no-context", False, False), ("zero-shot-image-first", True, True)],
)
def test_the_eval_script_writes_an_exp_row_for_each_zero_shot_predictor(
    tmp_path, data_dir, monkeypatch, name, context, image_first
):
    monkeypatch.setattr(model_eval, "git_state", lambda: ("0" * 40, False))
    backend = FakeBackend(json.dumps(GOOD))
    monkeypatch.setattr(apertus, "backend", lambda *args, **kwargs: backend)
    train = train_table()
    train["image_file"] = [str(image_file(data_dir, image_id)) for image_id in train["image_id"]]
    data = fake_data_lane(train, tmp_path / "out", FakeMetric())
    row = model_eval.run_eval(name, data, results_path=tmp_path / "results" / "metrics.json", exp_id="EXP-0002")
    assert row["model"] == "apertus-test"
    assert row["config"]["context"] is context
    assert row["config"]["image_first"] is image_first
    assert row["config"]["backend"] == "fake"
    assert row["config"]["image_size"] == 512
    assert len(row["config"]["prompt_sha256"]) == 16
    assert row["config"]["images"] == row["holdout"]["images"] == len(backend.calls)


def test_confidence_is_the_product_of_the_probabilities_of_the_tokens_that_spell_the_class():
    pieces = [('{"subject": "', 0.0), ("subj", -0.2), ("ect_b", -0.3), ('", "background": "', 0.0)]
    pieces += [('background_a"', -0.5), ("}", 0.0)]
    tokens = [{"token": token, "logprob": value} for token, value in pieces]
    labels = {"subject": "subject_b", "background": "background_a", "perspective": "perspective_a"}
    confidence = value_confidence(tokens, labels)
    assert confidence["subject"] == pytest.approx(math.exp(-0.5))
    assert confidence["background"] == pytest.approx(math.exp(-0.5)), "a token that also holds the quote counts"
    assert "perspective" not in confidence, "a class the tokens do not spell gets no number"
    assert value_confidence(None, labels) == {}
    assert value_confidence([{"token": "x"}], labels) == {}
    tokens[1]["logprob"] = float("nan")
    assert "subject" not in value_confidence(tokens, labels)


def test_the_predictor_fills_confidence_from_logprobs_and_counts_cells_without_one(data_dir):
    zero_shot = predictor(FakeBackend(with_logprobs(GOOD, -0.1)))
    [prediction] = zero_shot.predict([item(data_dir)])
    assert prediction.labels == GOOD
    assert prediction.confidence == pytest.approx({dimension: math.exp(-0.1) for dimension in DIMENSIONS})
    assert zero_shot.config()["no_confidence_cells"] == 0
    plain = predictor(FakeBackend(json.dumps(GOOD)))
    [prediction] = plain.predict([item(data_dir)])
    assert prediction.labels == GOOD and prediction.confidence is None
    assert plain.config()["no_confidence_cells"] == len(DIMENSIONS)


def test_after_a_retry_each_confidence_comes_from_the_answer_its_class_came_from(data_dir):
    first = with_logprobs({dimension: GOOD[dimension] for dimension in DIMENSIONS[:3]}, -0.1)
    second = with_logprobs({dimension: FAKE_CLASSES[dimension][1] for dimension in DIMENSIONS[:5]}, -0.7)
    zero_shot = predictor(FakeBackend(first, second))
    [prediction] = zero_shot.predict([item(data_dir)])
    assert [prediction.labels[d] for d in DIMENSIONS[:3]] == [GOOD[d] for d in DIMENSIONS[:3]]
    assert prediction.labels[DIMENSIONS[5]] == DEFAULTS[DIMENSIONS[5]]
    expected = {d: math.exp(-0.1) for d in DIMENSIONS[:3]} | {d: math.exp(-0.7) for d in DIMENSIONS[3:5]}
    assert prediction.confidence == pytest.approx(expected), "a default has no confidence"
    assert (zero_shot.config()["defaulted_cells"], zero_shot.config()["no_confidence_cells"]) == (1, 0)


def test_holdout_jsonl_carries_the_confidences_the_label_error_finder_reads(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(model_eval, "git_state", lambda: ("0" * 40, False))
    asked = {}
    backend = FakeBackend(with_logprobs(GOOD, -0.1))
    monkeypatch.setattr(apertus, "backend", lambda *args, **kwargs: asked.update(kwargs) or backend)
    train = train_table()
    train["image_file"] = [str(image_file(data_dir, image_id)) for image_id in train["image_id"]]
    data = fake_data_lane(train, tmp_path / "out", FakeMetric())
    row = model_eval.run_eval("zero-shot", data, results_path=tmp_path / "results" / "metrics.json", exp_id="EXP-0002")
    assert asked["extra_body"] == {"logprobs": True}
    frame = out_of_fold_frame(tmp_path / "out" / "predictions" / "EXP-0002")
    assert len(frame) == row["holdout"]["images"] * len(DIMENSIONS)
    assert list(frame["confidence"]) == pytest.approx([math.exp(-0.1)] * len(frame))
    assert set(frame["prediction"]) == set(GOOD.values())
