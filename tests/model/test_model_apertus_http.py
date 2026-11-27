"""Issue #13: the gateway and vLLM clients, against a fake transport: no network in ci."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image as PillowImage

from galaxus.model.apertus import BACKENDS, backend
from galaxus.model.apertus.api import GatewayBackend, gateway_settings, read_env_file
from galaxus.model.apertus.backend import APERTUS_8B, BackendError, data_url, image_bytes, pil_image
from galaxus.model.apertus.chat_http import ChatCompletionsBackend, chat_completions_url
from galaxus.model.apertus.vllm import DEFAULT_BASE_URL, VllmBackend

SECRET = "sk-test-not-a-real-key"  # secret-scan: allow (a fake key: the tests check it is never printed)
HI = [{"role": "user", "content": "hi"}]


def completion(text, prompt_tokens=42, completion_tokens=3, model="apertus-test") -> bytes:
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()


class FakeTransport:
    """Answers with the given (status, body) pairs in turn, repeating the last, and records every request."""

    def __init__(self, *answers):
        self.answers = list(answers) or [(200, completion("ok"))]
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def jpeg(size=(64, 32)) -> bytes:
    buffer = io.BytesIO()
    PillowImage.new("RGB", size, (200, 10, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def chat(transport, **options) -> ChatCompletionsBackend:
    return ChatCompletionsBackend(
        "https://gateway.example/v1", "apertus-test", SECRET, transport=transport, backoff=0, **options
    )


def test_images_go_before_the_text_of_the_last_user_message():
    transport = FakeTransport()
    messages = [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "this one"},
    ]
    chat(transport).generate(messages, [jpeg(), jpeg()])
    sent = transport.requests[0]["body"]["messages"]
    assert sent[:3] == messages[:3]
    parts = sent[3]["content"]
    assert [part["type"] for part in parts] == ["image_url", "image_url", "text"]
    assert parts[2]["text"] == "this one"
    header, encoded = parts[0]["image_url"]["url"].split(",", 1)
    assert header == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == jpeg()
    assert messages[3]["content"] == "this one", "the caller's messages are not changed"


def test_a_request_is_greedy_and_short_by_default():
    transport = FakeTransport()
    chat(transport).generate(HI)
    body = transport.requests[0]["body"]
    assert (body["model"], body["temperature"], body["max_tokens"]) == ("apertus-test", 0.0, 256)
    assert "response_format" not in body and "guided_json" not in body


@pytest.mark.parametrize("mode", ["response_format", "guided_json"])
def test_a_json_schema_travels_the_chosen_way(mode):
    schema = {"type": "object", "properties": {"color": {"type": "string", "enum": ["red"]}}}
    transport = FakeTransport()
    chat(transport, schema_mode=mode).generate(HI, json_schema=schema)
    body = transport.requests[0]["body"]
    if mode == "response_format":
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == schema
        assert "guided_json" not in body
    else:
        assert body["guided_json"] == schema
        assert "response_format" not in body


def test_schema_mode_none_sends_no_schema_and_a_bad_mode_is_refused():
    transport = FakeTransport()
    chat(transport, schema_mode="none").generate(HI, json_schema={"type": "object"})
    assert {"response_format", "guided_json"}.isdisjoint(transport.requests[0]["body"])
    with pytest.raises(ValueError, match="schema_mode"):
        chat(transport, schema_mode="grammar")


def test_extra_body_goes_into_every_request():
    transport = FakeTransport()
    chat(transport, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).generate(HI)
    assert transport.requests[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_answer_and_token_counts_are_read():
    transport = FakeTransport((200, completion("Red.", prompt_tokens=300, completion_tokens=2)))
    answer = chat(transport).generate(HI)
    assert (answer.text, answer.prompt_tokens, answer.completion_tokens) == ("Red.", 300, 2)
    assert (answer.finish_reason, answer.model) == ("stop", "apertus-test")
    assert answer.seconds >= 0


def test_an_answer_in_parts_is_joined():
    body = json.dumps({"choices": [{"message": {"content": [{"type": "text", "text": "Re"}, {"text": "d"}]}}]})
    assert chat(FakeTransport((200, body.encode()))).generate(HI).text == "Red"


def test_the_key_goes_only_into_the_authorization_header():
    transport = FakeTransport()
    client = chat(transport)
    client.generate(HI)
    request = transport.requests[0]
    assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in json.dumps(request["body"])
    assert SECRET not in repr(client)


def test_an_http_error_keeps_its_status_and_is_not_retried():
    transport = FakeTransport((400, b'{"error": "image input is not supported"}'))
    with pytest.raises(BackendError, match="HTTP 400.*not supported") as caught:
        chat(transport).generate(HI)
    assert caught.value.status == 400
    assert len(transport.requests) == 1


def test_a_busy_server_is_tried_again():
    transport = FakeTransport((503, b"busy"), (200, completion("ok")))
    assert chat(transport).generate(HI).text == "ok"
    assert len(transport.requests) == 2


def test_a_server_that_never_answers_is_a_backend_error():
    transport = FakeTransport(OSError("connection refused"))
    with pytest.raises(BackendError, match="no answer from the server"):
        chat(transport, retries=1).generate(HI)
    assert len(transport.requests) == 2


def test_an_answer_that_is_not_a_chat_completion_is_a_backend_error():
    with pytest.raises(BackendError, match="not a chat completion"):
        chat(FakeTransport((200, b"<html>sign in</html>"))).generate(HI)


def test_images_need_a_user_message():
    with pytest.raises(ValueError, match="user message"):
        chat(FakeTransport()).generate([{"role": "system", "content": "x"}], [jpeg()])


@pytest.mark.parametrize(
    "base",
    ["https://host.example/v1", "https://host.example/v1/", "https://host.example/v1/chat/completions"],
)
def test_the_endpoint_is_under_the_base_url(base):
    assert chat_completions_url(base) == "https://host.example/v1/chat/completions"


def test_a_base_url_without_a_scheme_is_refused():
    with pytest.raises(ValueError, match="http"):
        chat_completions_url("host.example/v1")


def test_bytes_and_files_pass_unchanged_and_pil_images_become_jpeg(tmp_path):
    png = io.BytesIO()
    PillowImage.new("RGB", (8, 8)).save(png, format="PNG")
    assert image_bytes(png.getvalue()) == (png.getvalue(), "image/png")
    path = tmp_path / "prepared.jpg"
    path.write_bytes(jpeg())
    assert image_bytes(path) == (jpeg(), "image/jpeg")
    encoded, mime = image_bytes(PillowImage.new("RGB", (8, 8)))
    assert mime == "image/jpeg" and encoded[:3] == b"\xff\xd8\xff"
    assert data_url(jpeg()).startswith("data:image/jpeg;base64,")
    assert pil_image(jpeg()).mode == "RGB"


def test_unknown_bytes_are_refused():
    with pytest.raises(ValueError, match="JPEG, PNG or WebP"):
        image_bytes(b"GIF89a not supported")


def test_a_transparent_image_is_refused_because_it_would_turn_black():
    transparent = PillowImage.new("RGBA", (8, 8), (0, 0, 0, 0))
    png = io.BytesIO()
    transparent.save(png, format="PNG")
    for image in (transparent, png.getvalue()):
        with pytest.raises(ValueError, match="onto white"):
            pil_image(image)
    with pytest.raises(ValueError, match="onto white"):
        image_bytes(transparent)


def test_gateway_settings_come_from_the_environment_before_the_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_NAME=from-file\nLLM_BASE_URL=https://file.example/v1\nLLM_API_KEY=file-key\n")
    assert gateway_settings({"LLM_NAME": "from-env"}, env_file) == {
        "LLM_NAME": "from-env",
        "LLM_BASE_URL": "https://file.example/v1",
        "LLM_API_KEY": "file-key",
    }


def test_the_env_file_allows_comments_export_and_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# the gateway\n"
        'export LLM_NAME="apertus-test"\n'
        "\n"
        "LLM_BASE_URL = 'https://g.example/v1'\n"
        "not a setting\n"
        "#LLM_API_KEY=commented-out\n"
    )
    assert read_env_file(env_file) == {"LLM_NAME": "apertus-test", "LLM_BASE_URL": "https://g.example/v1"}


def test_missing_gateway_settings_are_named_and_no_value_is_shown(tmp_path):
    with pytest.raises(BackendError) as caught:
        gateway_settings({"LLM_API_KEY": SECRET}, tmp_path / "missing.env")
    message = str(caught.value)
    assert "LLM_NAME" in message and "LLM_BASE_URL" in message
    assert SECRET not in message


def test_the_gateway_backend_is_built_from_the_llm_settings(tmp_path):
    transport = FakeTransport()
    settings = {"LLM_NAME": "apertus-test", "LLM_BASE_URL": "https://g.example/v1", "LLM_API_KEY": SECRET}
    gateway = GatewayBackend.from_env(settings, tmp_path / "missing.env", transport=transport)
    gateway.generate(HI)
    assert (gateway.name, gateway.model) == ("api", "apertus-test")
    assert transport.requests[0]["url"] == "https://g.example/v1/chat/completions"
    assert transport.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_vllm_defaults_to_a_local_server_with_the_8b_and_no_key():
    transport = FakeTransport()
    server = VllmBackend(transport=transport)
    server.generate(HI)
    assert (server.name, server.model) == ("vllm", APERTUS_8B)
    assert transport.requests[0]["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert "Authorization" not in transport.requests[0]["headers"]


def test_the_factory_builds_each_http_backend_and_refuses_unknown_names(monkeypatch):
    for key, value in {"LLM_NAME": "apertus-test", "LLM_BASE_URL": "https://g.example/v1", "LLM_API_KEY": SECRET}.items():
        monkeypatch.setenv(key, value)
    assert isinstance(backend("api"), GatewayBackend)
    assert isinstance(backend("vllm"), VllmBackend)
    assert set(BACKENDS) == {"api", "vllm", "local"}
    with pytest.raises(ValueError, match="unknown backend"):
        backend("closed-model")


# Issue #38: the gateway client goes through galaxus.keys.KeyRing.


def ring_env(**extra):
    return {"LLM_NAME": "apertus-test", "LLM_BASE_URL": "https://g.example/v1", "LLM_API_KEY": "key-one", **extra}


def ring_gateway(transport, tmp_path, **extra) -> GatewayBackend:
    return GatewayBackend.from_env(ring_env(**extra), tmp_path / "missing.env", transport=transport, backoff=0)


def keys_sent(transport) -> list[str]:
    return [request["headers"]["Authorization"].removeprefix("Bearer ") for request in transport.requests]


def test_a_402_on_key_1_moves_the_call_to_key_2(tmp_path):
    transport = FakeTransport((402, b'{"error": "out of credits"}'), (200, completion("ok")))
    gateway = ring_gateway(transport, tmp_path, LLM_API_KEY_2="key-two")
    assert gateway.generate(HI).text == "ok"
    assert keys_sent(transport) == ["key-one", "key-two"]
    gateway.generate(HI)
    assert keys_sent(transport)[-1] == "key-two", "later calls stay on the key that works"


def test_a_401_also_moves_to_the_next_key(tmp_path):
    transport = FakeTransport((401, b'{"error": "invalid key"}'), (200, completion("ok")))
    assert ring_gateway(transport, tmp_path, LLM_API_KEY_2="key-two").generate(HI).text == "ok"
    assert keys_sent(transport) == ["key-one", "key-two"]


def test_when_every_key_is_spent_no_key_is_shown(tmp_path, capsys):
    transport = FakeTransport((402, b'{"error": "out of credits"}'))
    with pytest.raises(RuntimeError, match="all 2 API keys") as caught:
        ring_gateway(transport, tmp_path, LLM_API_KEY_2="key-two").generate(HI)
    shown = str(caught.value) + str(caught.value.__cause__) + capsys.readouterr().err
    assert "key-one" not in shown and "key-two" not in shown
    assert "key 1" in shown, "the switch is logged by position"


def test_a_busy_server_is_retried_on_the_same_key(tmp_path):
    transport = FakeTransport((503, b"busy"), (200, completion("ok")))
    assert ring_gateway(transport, tmp_path, LLM_API_KEY_2="key-two").generate(HI).text == "ok"
    assert keys_sent(transport) == ["key-one", "key-one"]


def test_keys_come_from_the_env_file_and_the_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY_2=file-two\nLLM_API_KEY_3=file-three\n")
    gateway = GatewayBackend.from_env(ring_env(LLM_API_KEY_3="env-three", LLM_API_KEY_4=" "), env_file)
    assert gateway.keys.keys == ["key-one", "file-two", "env-three"]


def test_the_repr_counts_the_keys_and_shows_none(tmp_path):
    gateway = ring_gateway(FakeTransport(), tmp_path, LLM_API_KEY_2="key-two")
    assert "keys=2" in repr(gateway)
    assert "key-one" not in repr(gateway) and "key-two" not in repr(gateway)
