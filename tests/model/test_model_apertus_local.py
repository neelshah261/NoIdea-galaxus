"""Issue #13: the local backend, with stand-ins for torch and transformers: no weights, no GPU, no network."""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image as PillowImage

from galaxus.model.apertus.backend import APERTUS_8B, BackendError
from galaxus.model.apertus.local import LocalBackend, pick_device

HIDDEN = 8
LAYERS = 3
ANSWER_TOKENS = [100, 101, 102]
PROMPT_TOKENS = 5
TOKENS_PER_IMAGE = 4


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    """Counts ``PROMPT_TOKENS`` plus ``TOKENS_PER_IMAGE`` per image and records the chats it formats."""

    @classmethod
    def from_pretrained(cls, model, **options):
        return cls()

    def __init__(self):
        self.chats = []

    def apply_chat_template(self, chat, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        self.chats.append(chat)
        return "PROMPT"

    def __call__(self, text, images=None, return_tensors=None):
        length = PROMPT_TOKENS + TOKENS_PER_IMAGE * len(images or [])
        return FakeInputs(input_ids=np.arange(length).reshape(1, length))

    def decode(self, tokens, skip_special_tokens):
        return " ".join(str(token) for token in tokens)


class FakeNetwork:
    """Generates ``ANSWER_TOKENS``; hidden state of layer ``l`` at token ``t`` is ``100 * l + t`` in every unit."""

    loaded_with: ClassVar[dict] = {}

    @classmethod
    def from_pretrained(cls, model, **options):
        cls.loaded_with = {"model": model, **options}
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, max_new_tokens, **sampling):
        self.generated_with = {"max_new_tokens": max_new_tokens, **sampling}
        return np.concatenate([input_ids, np.array([ANSWER_TOKENS])], axis=1)

    def __call__(self, input_ids, output_hidden_states):
        tokens = input_ids.shape[1]
        position = np.arange(tokens).reshape(1, tokens, 1)
        return SimpleNamespace(
            hidden_states=tuple(np.zeros((1, tokens, HIDDEN)) + 100 * layer + position for layer in range(LAYERS))
        )


@pytest.fixture
def stack(monkeypatch):
    """Stand-in torch and transformers modules; a test sets which devices torch reports."""
    torch = types.ModuleType("torch")
    torch.__version__ = "9.9.0"
    torch.devices = {"cuda": False, "mps": False}
    torch.cuda = SimpleNamespace(is_available=lambda: torch.devices["cuda"])
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: torch.devices["mps"]))
    torch.inference_mode = contextlib.nullcontext
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "5.24.0"
    transformers.AutoProcessor = FakeProcessor
    transformers.AutoModelForImageTextToText = FakeNetwork
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return torch, transformers


def red(size=(16, 16)):
    return PillowImage.new("RGB", size, (200, 0, 0))


def test_it_loads_apertus_with_the_image_text_class(stack):
    local = LocalBackend()
    assert (local.model, local.device, local.dtype) == (APERTUS_8B, "cpu", "float32")
    assert local.model_class is FakeNetwork
    assert FakeNetwork.loaded_with == {"model": APERTUS_8B, "revision": None, "trust_remote_code": False, "dtype": "fp32"}
    assert (local.transformers_version, local.torch_version) == ("5.24.0", "9.9.0")
    assert local.load_seconds >= 0


def test_transformers_before_4_56_get_torch_dtype(stack):
    stack[1].__version__ = "4.55.2"
    LocalBackend()
    assert FakeNetwork.loaded_with["torch_dtype"] == "fp32"
    assert "dtype" not in FakeNetwork.loaded_with


@pytest.mark.parametrize(
    ("cuda", "mps", "device", "dtype"),
    [(True, True, "cuda", "bfloat16"), (False, True, "mps", "float16"), (False, False, "cpu", "float32")],
)
def test_the_best_device_is_picked_with_a_dtype_that_suits_it(stack, cuda, mps, device, dtype):
    torch, _ = stack
    torch.devices.update(cuda=cuda, mps=mps)
    assert pick_device(torch) == device
    local = LocalBackend()
    assert (local.device, local.dtype) == (device, dtype)


def test_an_explicit_device_and_dtype_win(stack):
    stack[0].devices["cuda"] = True
    local = LocalBackend(device="cpu", dtype="bfloat16")
    assert (local.device, local.dtype) == ("cpu", "bfloat16")
    assert FakeNetwork.loaded_with["dtype"] == "bf16"


def test_generate_decodes_only_the_new_tokens(stack):
    local = LocalBackend()
    answer = local.generate([{"role": "user", "content": "what color?"}], [red(), red()], max_tokens=7)
    assert answer.text == "100 101 102"
    assert (answer.prompt_tokens, answer.completion_tokens) == (PROMPT_TOKENS + 2 * TOKENS_PER_IMAGE, 3)
    assert local.network.generated_with == {"max_new_tokens": 7, "do_sample": False}
    (chat,) = local.processor.chats
    assert chat[-1]["content"] == [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "what color?"}]


def test_sampling_is_on_only_with_a_temperature(stack):
    local = LocalBackend()
    local.generate([{"role": "user", "content": "hi"}], temperature=0.7)
    assert local.network.generated_with == {"max_new_tokens": 256, "do_sample": True, "temperature": 0.7}


def test_forward_gives_one_feature_per_image_and_layer(stack):
    local = LocalBackend()
    tokens = PROMPT_TOKENS + TOKENS_PER_IMAGE
    last = local.forward([red(), red()], "describe", layers=(-1, 0))
    assert last.shape == (2, 2, HIDDEN)
    assert last[0, 0, 0] == 100 * (LAYERS - 1) + tokens - 1, "last layer, last token"
    assert last[1, 1, 0] == tokens - 1, "first layer, last token"
    mean = local.forward([red()], "describe", pool="mean")
    assert mean.shape == (1, 1, HIDDEN)
    assert mean[0, 0, 0] == pytest.approx(100 * (LAYERS - 1) + (tokens - 1) / 2)


def test_forward_of_no_images_is_empty_and_an_unknown_pool_is_refused(stack):
    local = LocalBackend()
    assert local.forward([], "describe").shape == (0, 1, 0)
    with pytest.raises(ValueError, match="pool"):
        local.forward([red()], "describe", pool="max")


def test_a_model_that_does_not_load_names_the_transformers_version(stack, monkeypatch):
    def gated(model, **options):
        raise OSError("gated repo: accept the terms on Hugging Face")

    monkeypatch.setattr(FakeNetwork, "from_pretrained", staticmethod(gated))
    with pytest.raises(BackendError, match="transformers 5.24.0 could not load .*gated repo"):
        LocalBackend()


def test_without_torch_the_error_says_how_to_run_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BackendError, match="uv run --with torch --with transformers"):
        LocalBackend()


class CueTokenizer:
    """One token per character of a cue, numbered from 1000, so a cue's token count is its text's length."""

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [1000 + index for index in range(len(text))]}


def test_fields_reads_the_prompt_end_its_mean_and_each_cue_end_in_one_pass(stack):
    """#214: per layer, the "last" and "mean" features, then where Apertus would write each field's class."""
    local = LocalBackend()
    local.processor.tokenizer = CueTokenizer()
    tokens = PROMPT_TOKENS + TOKENS_PER_IMAGE
    got = local.forward([red()], "describe", layers=(-1, 0), pool="fields", fields=("a", "bb"))
    assert got.shape == (1, 2 * 4, HIDDEN), "two layers x (last, mean, 2 cues)"
    top = 100 * (LAYERS - 1)
    first_cue, second_cue = len("a:"), len("\nbb:")
    assert got[0, 0, 0] == top + tokens - 1, "the prompt's last token, as pool='last'"
    assert got[0, 1, 0] == pytest.approx(top + (tokens - 1) / 2), "the prompt's mean, as pool='mean'"
    assert got[0, 2, 0] == top + tokens + first_cue - 1, "the end of the first cue"
    assert got[0, 3, 0] == top + tokens + first_cue + second_cue - 1, "the end of the second cue"
    assert got[0, 4, 0] == tokens - 1, "the next layer starts with its own last prompt token"
    last = local.forward([red()], "describe", layers=(-1, 0))
    assert np.array_equal(got[0, [0, 4]], last[0]), "the 'last' features are kept exactly"


def test_fields_extends_the_attention_mask_and_refuses_inputs_it_does_not_know(stack):
    local = LocalBackend()
    local.processor.tokenizer = CueTokenizer()
    inputs = {"input_ids": np.arange(9).reshape(1, 9), "attention_mask": np.ones((1, 9), dtype=int),
              "pixel_values": np.zeros((1, 3, 4, 4))}
    extended, ends = local._with_cues(inputs, ("a",))
    assert ends == [8, 10] and extended["input_ids"].shape == (1, 11)
    assert extended["attention_mask"].tolist() == [[1] * 11]
    assert extended["pixel_values"] is inputs["pixel_values"], "anything not shaped like the ids is left as it is"
    with pytest.raises(BackendError, match="does not know how to extend"):
        local._with_cues({**inputs, "position_hint": np.zeros((1, 9))}, ("a",))
    with pytest.raises(ValueError, match="field names"):
        local.forward([red()], "describe", pool="fields")
