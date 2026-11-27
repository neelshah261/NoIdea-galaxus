"""train/lora_apertus.py's training conversation has the shape Apertus 1.5's chat template accepts (#76).

The template itself is the model's (gated, not in this repo); its rules, checked against it on 27 Nov: the system
and assistant content must be strings, the user content a list of parts. A list for the system raised
"Invalid system message" on the first CPU rehearsal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "train" / "lora_apertus.py"
spec = importlib.util.spec_from_file_location("galaxus_train_lora_apertus", PATH)
la = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = la
spec.loader.exec_module(la)


def test_system_and_answer_are_strings_and_the_image_comes_first():
    image = object()
    system, user, assistant = la.chat_messages("the instruction", image, {"product_title": "Mouse"}, '{"a": "b"}')
    assert system == {"role": "system", "content": "the instruction"}
    assert assistant == {"role": "assistant", "content": '{"a": "b"}'}
    assert user["role"] == "user" and [p["type"] for p in user["content"]] == ["image", "text"]
    assert user["content"][0]["image"] is image and "Mouse" in user["content"][1]["text"]


def test_no_context_gives_the_inference_prompt():
    _, user, _ = la.chat_messages("s", object(), None, "{}")
    assert user["content"][1]["text"] == la.user_prompt(None)
