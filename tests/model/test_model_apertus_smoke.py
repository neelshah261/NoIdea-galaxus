"""Issue #13: the smoke script's checks and its "For design" block, against a fake backend: no network."""

from __future__ import annotations

import json

import numpy as np

from galaxus.model.apertus import smoke
from galaxus.model.apertus.backend import BackendError, Generation
from galaxus.model.apertus.chat_http import ChatCompletionsBackend
from galaxus.model.apertus.smoke import (
    check_json_schema,
    check_logprobs,
    conforms,
    pixels_per_token,
    render_markdown,
    run_smoke,
    square,
)

TEXT_TOKENS = 20


class FakeBackend:
    """A server whose behaviour each test sets: images, token counting, which schema mode it enforces."""

    name = "fake"
    model = "apertus-fake"

    def __init__(
        self, *, images=True, sees=True, pixels=32, enforced=("response_format",), refused=(), think=False, fixed=None
    ):
        self.schema_mode = "response_format"
        self.images, self.sees, self.pixels, self.think, self.fixed = images, sees, pixels, think, fixed
        self.enforced, self.refused = enforced, refused

    def generate(self, messages, images=(), json_schema=None, *, max_tokens=256, temperature=0.0):
        if images and not self.images:
            raise BackendError("fake: HTTP 400: image input is not supported", status=400)
        if json_schema is not None and self.schema_mode in self.refused:
            raise BackendError(f"fake: HTTP 400: unknown field {self.schema_mode}", status=400)
        if json_schema is not None:
            text = json.dumps({"color": "red", "shape": "square"}) if self.schema_mode in self.enforced else "hello"
        elif images:
            text = "Red." if self.sees else "A cat."
        else:
            text = "There is no image."
        if self.think:
            text = f"<think>hmm</think>{text}"
        image_tokens = sum(self.fixed or (image.width // self.pixels) * (image.height // self.pixels) for image in images)
        return Generation(text=text, prompt_tokens=TEXT_TOKENS + image_tokens, completion_tokens=2, seconds=0.5)


def test_a_backend_that_sees_images_answers_q_0001_and_q_0002():
    report = run_smoke(FakeBackend())
    assert report["text_only"]["ok"]
    assert report["image"]["taken"] and report["image"]["seen"]
    assert report["image_tokens"]["per_size"] == {"256x256": 64, "512x512": 256, "768x768": 576, "512x256": 128}
    assert (report["image_tokens"]["side"], report["image_tokens"]["proportional"]) == (32.0, True)
    assert report["json_schema"] == {"response_format": "enforced"}
    assert report["think_block"] is False
    assert report["local"] is None
    block = render_markdown(report)
    assert "**Q-0001 images:** yes, and the model sees it" in block
    assert "about one token per 32 x 32 px" in block
    assert "response_format: enforced" in block
    assert "run `--backend local`" in block


def test_a_backend_without_images_says_no_and_counts_no_tokens():
    report = run_smoke(FakeBackend(images=False))
    assert not report["image"]["taken"]
    assert report["image_tokens"]["per_size"] == {}
    assert "**Q-0001 images:** no: fake: HTTP 400: image input is not supported" in render_markdown(report)


def test_an_image_that_is_taken_but_not_seen_is_flagged():
    assert "taken, but the answer to a red square was 'A cat.'" in render_markdown(run_smoke(FakeBackend(sees=False)))


def test_guided_json_is_tried_when_response_format_is_refused():
    backend = FakeBackend(enforced=("guided_json",), refused=("response_format",))
    report = run_smoke(backend)
    assert report["json_schema"]["response_format"].startswith("refused: fake: HTTP 400")
    assert report["json_schema"]["guided_json"] == "enforced"
    assert backend.schema_mode == "response_format", "the backend's own mode is put back"


def test_a_schema_that_is_ignored_is_reported():
    assert run_smoke(FakeBackend(enforced=()))["json_schema"] == {
        "response_format": "ignored: answered 'hello'",
        "guided_json": "ignored: answered 'hello'",
    }


def test_a_backend_without_schema_modes_enforces_none():
    class InProcess:
        name, model = "local", "apertus-fake"

    assert list(check_json_schema(InProcess(), [])) == ["local"]


def test_tokens_that_do_not_grow_with_the_area_are_reported():
    report = run_smoke(FakeBackend(fixed=729))
    assert report["image_tokens"]["proportional"] is False
    assert "not proportional to the area" in render_markdown(report)


def test_a_think_block_is_noticed():
    assert run_smoke(FakeBackend(think=True))["think_block"] is True


def test_a_local_backend_reports_what_loaded_it():
    class Local(FakeBackend):
        name = "local"
        transformers_version, torch_version, device, dtype, load_seconds = "5.24.0", "2.9.0", "mps", "float16", 12.5
        model_class = type("AutoModelForImageTextToText", (), {})

        def forward(self, images, prompt):
            return np.zeros((len(images), 1, 4096))

    report = run_smoke(Local())
    assert report["local"]["hidden_states"] == {"shape": [1, 1, 4096]}
    assert (
        "transformers 5.24.0 (torch 2.9.0) loaded apertus-fake with AutoModelForImageTextToText on mps"
        in render_markdown(report)
    )


def test_a_backend_that_does_not_answer_at_all_fails_the_run(monkeypatch, capsys):
    class Down(FakeBackend):
        def generate(self, *args, **kwargs):
            raise BackendError("fake: no answer from the server: connection refused")

    monkeypatch.setattr(smoke, "build_backend", lambda args: Down())
    assert smoke.main(["--backend", "vllm"]) == 1
    assert "did not answer at all" in capsys.readouterr().out


def test_the_command_line_prints_the_block_and_writes_the_report(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smoke, "build_backend", lambda args: FakeBackend())
    target = tmp_path / "out" / "smoke.json"
    assert smoke.main(["--backend", "api", "--json", str(target)]) == 0
    assert "B6 smoke test (#13)" in capsys.readouterr().out
    assert json.loads(target.read_text(encoding="utf-8"))["image"]["seen"] is True


def test_missing_settings_end_the_command_line_with_the_reason(monkeypatch, capsys):
    def no_settings(args):
        raise BackendError("api: LLM_NAME not set")

    monkeypatch.setattr(smoke, "build_backend", no_settings)
    assert smoke.main(["--backend", "api"]) == 1
    assert "LLM_NAME not set" in capsys.readouterr().err


def test_conforms_needs_exactly_the_schema():
    assert conforms('{"color": "red", "shape": "square"}')
    assert not conforms('{"color": "red"}')
    assert not conforms('{"color": "pink", "shape": "square"}')
    assert not conforms("hello")


def test_pixels_per_token_needs_two_sizes():
    assert pixels_per_token({"512x512": 256}) == {"side": None, "proportional": None}
    assert pixels_per_token({"512x512": 256, "256x256": 64}) == {"side": 32.0, "proportional": True}


def test_the_square_is_red_on_white():
    image = square((100, 50))
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((50, 25)) == (220, 30, 30)


def server(sends_logprobs: bool) -> ChatCompletionsBackend:
    """A chat server that answers schema JSON, with logprobs when asked if ``sends_logprobs``."""

    def transport(url, headers, body, timeout):
        asked = json.loads(body).get("logprobs") and sends_logprobs
        logprobs = {"content": [{"token": "red", "logprob": 0.0}]} if asked else None
        choice = {"message": {"content": json.dumps({"color": "red", "shape": "square"})}, "logprobs": logprobs}
        return 200, json.dumps({"choices": [choice]}).encode()

    return ChatCompletionsBackend("https://gateway.example/v1", "apertus-test", transport=transport, backoff=0)


def test_the_block_says_whether_a_schema_answer_carries_logprobs():
    backend = server(sends_logprobs=True)
    assert check_logprobs(backend) is True
    assert backend.extra_body == {}, "the check leaves the backend as it was"
    assert check_logprobs(server(sends_logprobs=False)) is False
    assert check_logprobs(FakeBackend()) is None
    assert "logprobs with the JSON schema (#75):** not asked" in render_markdown(run_smoke(FakeBackend()))
