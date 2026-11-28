"""``local``: Apertus v1.5 in this process with transformers, on cuda, mps or cpu (issue #13).

The one backend with :meth:`LocalBackend.forward`, the hidden states that heads on frozen features need (Q-0007).

torch and transformers are not dependencies of the package yet: they are the ``local`` extra of B0 (#11). Until
then, add them for one run without touching the lock file:

    uv run --with torch --with transformers python -m galaxus.model.apertus.smoke --backend local

Which transformers version loads Apertus v1.5 is Q-0006 (the swiss-ai fork on the model card, or a release); the
smoke script reports the version and the model class it ran with. A GPU run is a human's call (AGENTS.md product
rule 7).
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from galaxus.model.apertus.backend import (
    APERTUS_8B,
    DEFAULT_MAX_TOKENS,
    BackendError,
    Generation,
    ImageInput,
    Message,
    last_user_index,
    pil_image,
)

#: Tried in order: the first class transformers has is used, and the smoke script reports which.
MODEL_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM")
#: ``dtype="auto"``: bf16 where it is fast, fp16 on Apple GPUs, fp32 on the CPU.
AUTO_DTYPES = {"cuda": "bfloat16", "mps": "float16", "cpu": "float32"}
#: ``fields`` (#214): one pass that also reads where Apertus would write each field's class, see ``forward``.
POOLS = ("last", "mean", "fields")
#: Inputs shaped like ``input_ids`` that ``fields`` knows how to extend for the cue tokens, with the value to append.
CUE_FILL = {"attention_mask": 1, "token_type_ids": 0, "mm_token_type_ids": 0}


def import_stack() -> tuple[Any, Any]:
    """torch and transformers, or an error that says how to get them."""
    try:
        import torch
        import transformers
    except ModuleNotFoundError as missing:
        raise BackendError(
            f"local: {missing.name} is not installed. It comes with the 'local' extra of B0 (#11); until then run "
            "with: uv run --with torch --with transformers python -m galaxus.model.apertus.smoke --backend local"
        ) from missing
    return torch, transformers


def pick_device(torch: Any) -> str:
    """cuda, else Apple's mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _version(text: str) -> tuple[int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", text)[:2]]
    return (numbers + [0, 0])[0], (numbers + [0, 0])[1]


def _numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _field_rows(hidden: Any, ends: Sequence[int]) -> list[np.ndarray]:
    """``pool="fields"`` of one layer (``hidden`` is tokens x units): the prompt's last token, the prompt's mean, then
    each cue's last token. Rows are picked on the device, so only a few vectors are copied, never the whole layer."""
    prompt_end = ends[0] + 1
    prompt = hidden[:prompt_end]
    mean = _numpy(prompt.float().mean(0)) if hasattr(prompt, "float") else np.asarray(prompt, np.float32).mean(axis=0)
    return [_numpy(hidden[ends[0]]), mean, *(_numpy(hidden[end]) for end in ends[1:])]


#: Where Apertus 1.5 keeps its VQ image tokenizer. It stays in float32 when the rest of the model runs in bf16 or fp16:
#: the training path does the same (``train/lora_apertus.py``, ``keep_vision_tokenizer_fp32``, from the vllm#50496
#: review), so the image codes here are the ones the LoRA run and the gateway see (#218).
VISION_TOKENIZER_NAMES = ("vision_tokenizer", "image_tokenizer")


def keep_vision_tokenizer_fp32(network: Any) -> int:
    """Cast every vision-tokenizer module of ``network`` to float32; returns how many there were (0 for a model
    without one, or a stand-in without ``named_modules``)."""
    named_modules = getattr(network, "named_modules", None)
    if not callable(named_modules):
        return 0
    found = [module for name, module in named_modules() if name.endswith(VISION_TOKENIZER_NAMES)]
    for module in found:
        module.float()
    return len(found)


def chat_with_images(messages: Sequence[Message], image_count: int) -> list[dict[str, Any]]:
    """``messages`` in the transformers chat format, with ``image_count`` image slots before the last user text."""
    chat = [{"role": message["role"], "content": [{"type": "text", "text": message["content"]}]} for message in messages]
    if image_count:
        index = last_user_index(chat)
        chat[index]["content"] = [{"type": "image"} for _ in range(image_count)] + chat[index]["content"]
    return chat


class LocalBackend:
    """Apertus loaded with transformers in this process."""

    name = "local"

    def __init__(
        self,
        model: str = APERTUS_8B,
        *,
        device: str | None = None,
        dtype: str = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.torch, self.transformers = import_stack()
        self.model = model
        self.revision = revision
        self.device = device or pick_device(self.torch)
        self.dtype = AUTO_DTYPES.get(self.device, "float32") if dtype == "auto" else dtype
        self.transformers_version = str(self.transformers.__version__)
        self.torch_version = str(getattr(self.torch, "__version__", "unknown"))
        self.model_class = next(
            (getattr(self.transformers, name) for name in MODEL_CLASSES if hasattr(self.transformers, name)), None
        )
        if self.model_class is None:
            raise BackendError(f"local: transformers {self.transformers_version} has none of {', '.join(MODEL_CLASSES)}")
        # transformers 4.56 renamed from_pretrained's torch_dtype to dtype.
        dtype_argument = "dtype" if _version(self.transformers_version) >= (4, 56) else "torch_dtype"
        loading: dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code}
        started = time.perf_counter()
        try:
            self.processor = self.transformers.AutoProcessor.from_pretrained(model, **loading)
            self.network = self.model_class.from_pretrained(
                model, **loading, **{dtype_argument: getattr(self.torch, self.dtype)}
            )
        except Exception as error:  # any loading failure is the Q-0006 answer, so report it whole
            raise BackendError(
                f"local: transformers {self.transformers_version} could not load {model} with "
                f"{self.model_class.__name__}: {type(error).__name__}: {error}"
            ) from error
        self.vision_tokenizers_fp32 = keep_vision_tokenizer_fp32(self.network)
        self.network.to(self.device)
        self.network.eval()
        self.load_seconds = time.perf_counter() - started

    def _inputs(self, messages: Sequence[Message], images: Sequence[ImageInput]) -> Mapping[str, Any]:
        pictures = [pil_image(image) for image in images]
        prompt = self.processor.apply_chat_template(
            chat_with_images(messages, len(pictures)), add_generation_prompt=True, tokenize=False
        )
        return self.processor(text=[prompt], images=pictures or None, return_tensors="pt").to(self.device)

    def generate(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput] = (),
        json_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Generation:
        """Answer in this process. ``json_schema`` is not enforced here: transformers has no constrained decoding
        built in, so parse the answer leniently (the ``vllm`` backend enforces the schema)."""
        inputs = self._inputs(messages, images)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.network.generate(**inputs, max_new_tokens=max_tokens, **sampling)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = output[0][prompt_tokens:]
        decoder = getattr(self.processor, "tokenizer", self.processor)
        return Generation(
            text=decoder.decode(new_tokens, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(new_tokens),
            model=self.model,
            seconds=time.perf_counter() - started,
        )

    def forward(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        layers: Sequence[int] = (-1,),
        pool: str = "last",
        fields: Sequence[str] = (),
    ) -> np.ndarray:
        """Hidden states of ``prompt`` with each image, shape ``(images, layers * blocks, hidden size)``.

        ``pool="last"`` takes the last token, the one the answer would start from; ``"mean"`` averages all tokens
        (one block per layer). ``pool="fields"`` (#214) appends one cue per field after the answer start, ``"<field>:"``
        on its own line, as Apertus would begin writing it, in the same pass. Per layer it gives ``2 + len(fields)``
        blocks: the last prompt token and the prompt's mean (the ``last`` and ``mean`` features), then each cue's last
        token, where Apertus would write that field's class. One image per pass, so a feature never depends on padding.
        """
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {', '.join(POOLS)}, got {pool!r}")
        if pool == "fields" and not fields:
            raise ValueError("pool 'fields' needs the field names (galaxus.data.classes.dimensions())")
        features = []
        for image in images:
            inputs = self._inputs([{"role": "user", "content": prompt}], [image])
            ends: list[int] = []
            if pool == "fields":
                inputs, ends = self._with_cues(inputs, fields)
            with self.torch.inference_mode():
                states = self.network(**inputs, output_hidden_states=True).hidden_states
            picked = []
            for layer in layers:
                if pool == "fields":
                    picked.extend(_field_rows(states[layer][0], ends))
                    continue
                hidden = _numpy(states[layer][0])
                picked.append(hidden[-1] if pool == "last" else hidden.mean(axis=0))
            features.append(np.stack(picked))
        if not features:
            return np.empty((0, len(layers), 0), dtype=np.float32)
        return np.stack(features)

    def _with_cues(self, inputs: Mapping[str, Any], fields: Sequence[str]) -> tuple[dict[str, Any], list[int]]:
        """``inputs`` with one cue per field appended, and the positions read: the prompt's last token, then each cue's
        last token. Each cue is tokenized on its own, so its end is exact whatever the tokenizer merges."""
        ids = inputs["input_ids"]
        length = int(ids.shape[1])
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        appended: list[int] = []
        ends = [length - 1]
        for index, field in enumerate(fields):
            cue = [int(token) for token in tokenizer(("\n" if index else "") + f"{field}:", add_special_tokens=False)["input_ids"]]
            if not cue:
                raise BackendError(f"local: the cue of field {field!r} has no tokens")
            appended.extend(cue)
            ends.append(length + len(appended) - 1)
        extended: dict[str, Any] = {}
        for key, value in inputs.items():
            shape = tuple(getattr(value, "shape", ()))
            if key == "input_ids":
                extended[key] = self._append(value, appended)
            elif len(shape) == 2 and shape == (1, length):
                if key not in CUE_FILL:
                    raise BackendError(f"local: pool 'fields' does not know how to extend the input {key!r}")
                extended[key] = self._append(value, [CUE_FILL[key]] * len(appended))
            else:
                extended[key] = value
        return extended, ends

    def _append(self, value: Any, tokens: Sequence[int]) -> Any:
        """``value`` (1 x n) with ``tokens`` appended along the sequence, as a numpy array or a tensor on its device."""
        if isinstance(value, np.ndarray):
            return np.concatenate([value, np.asarray([tokens], dtype=value.dtype)], axis=1)
        extra = self.torch.tensor([list(tokens)], dtype=value.dtype, device=value.device)
        return self.torch.cat([value, extra], dim=1)
