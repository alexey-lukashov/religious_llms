from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning


@dataclass
class ChatResult:
    message: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    finish_reason: Optional[str]
    usage: Dict[str, Any]
    cost: Optional[float]
    raw: Dict[str, Any]


def _parse_json_maybe(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if text is None:
        return {}
    if not isinstance(text, str):
        return {"value": text}
    try:
        return json.loads(text)
    except Exception:
        # Try to salvage simple cases
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        return {"raw": text}


def _request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    retries = kwargs.pop("retries", 2)
    backoff = kwargs.pop("backoff", 1.5)
    timeout = kwargs.pop("timeout", 60)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code < 500:
                return resp
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(backoff ** attempt)
    if last_exc:
        raise last_exc
    return resp  # type: ignore[misc]


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        token_param: str = "max_tokens",
        verify_ssl: bool = True,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = {k: v for k, v in (extra_headers or {}).items() if v}
        self.extra_body = extra_body or {}
        self.token_param = token_param
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        payload[self.token_param] = max_tokens
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.extra_body:
            for key, value in self.extra_body.items():
                if key not in payload:
                    payload[key] = value

        resp = _request_with_retries(
            "POST",
            url,
            headers=headers,
            json=payload,
            verify=self.verify_ssl,
            timeout=self.timeout_seconds,
        )
        try:
            data = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:200]
            raise RuntimeError(f"HTTP {resp.status_code}: non-JSON response: {snippet}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")

        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        tool_calls = message.get("tool_calls") or []
        if not tool_calls and message.get("function_call"):
            # Legacy single function call
            fc = message["function_call"]
            tool_calls = [
                {
                    "id": f"legacy_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": fc.get("name"), "arguments": fc.get("arguments")},
                }
            ]
        usage = data.get("usage") or {}
        cost = None
        for key in ("cost", "total_cost"):
            if key in data:
                try:
                    cost = float(data.get(key))
                except (TypeError, ValueError):
                    cost = None
                break
        if cost is None and isinstance(usage, dict) and "cost" in usage:
            try:
                cost = float(usage.get("cost"))
            except (TypeError, ValueError):
                cost = None
        return ChatResult(
            message=message,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            cost=cost,
            raw=data,
        )


class YandexClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        folder_id: str = "",
        verify_ssl: bool = True,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key, self.folder_id = self._parse_api_key(api_key, folder_id)
        self.model = model
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _parse_api_key(api_key: str, folder_id: str) -> Tuple[str, str]:
        if ":" in api_key:
            parts = api_key.rsplit(":", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]
        if not api_key:
            raise RuntimeError("Missing Yandex API key")
        if not folder_id:
            raise RuntimeError("Missing Yandex folder id")
        return api_key, folder_id

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                tool_name = msg.get("name", "tool")
                tool_content = content or ""
                converted.append(
                    {"role": "user", "text": f"[Tool Result for {tool_name}]: {tool_content}"}
                )
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_text_parts = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tool_text_parts.append(
                            f"[Calling {func.get('name', 'tool')}({func.get('arguments', '{}')})]"
                        )
                    text = ((content or "") + " " + " ".join(tool_text_parts)).strip()
                    if text:
                        converted.append({"role": "assistant", "text": text})
                    continue
                if content:
                    converted.append({"role": "assistant", "text": content})
                continue

            if isinstance(content, str):
                if content:
                    converted.append({"role": role, "text": content})
            elif isinstance(content, list):
                text_parts = []
                image_url = None
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img = part.get("image_url", {})
                        image_url = img.get("url") if isinstance(img, dict) else img
                text = " ".join(text_parts)
                if text or image_url:
                    yandex_msg: Dict[str, Any] = {"role": role, "text": text or "Image:"}
                    if image_url:
                        if isinstance(image_url, str) and image_url.startswith("data:"):
                            parts = image_url.split(",", 1)
                            if len(parts) == 2:
                                yandex_msg["image"] = {
                                    "imageData": {"mimeType": "image/jpeg", "data": parts[1]}
                                }
                        else:
                            yandex_msg["image"] = {"imageUrl": image_url}
                    converted.append(yandex_msg)
            elif content:
                converted.append({"role": role, "text": str(content)})

        return converted

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        endpoint = f"{self.base_url}/completion"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,
        }
        body: Dict[str, Any] = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {"stream": False, "temperature": temperature},
            "messages": self._convert_messages(messages),
        }
        if max_tokens:
            body["completionOptions"]["maxTokens"] = str(max_tokens)
        if tools:
            body["tools"] = tools

        resp = _request_with_retries(
            "POST",
            endpoint,
            headers=headers,
            json=body,
            verify=self.verify_ssl,
            timeout=self.timeout_seconds,
        )
        try:
            data = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:200]
            raise RuntimeError(f"HTTP {resp.status_code}: non-JSON response: {snippet}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")

        result = data.get("result", {})
        alternatives = result.get("alternatives", [])
        usage = result.get("usage") or data.get("usage") or {}

        openai_message: Dict[str, Any] = {"role": "assistant", "content": ""}
        tool_calls: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None

        if alternatives:
            alt = alternatives[0]
            message = alt.get("message", {})
            openai_message = {
                "role": message.get("role", "assistant"),
                "content": message.get("text", ""),
            }
            tool_call_list = message.get("toolCallList", {})
            y_tool_calls = tool_call_list.get("toolCalls", [])
            if y_tool_calls:
                for idx, tc in enumerate(y_tool_calls):
                    func_call = tc.get("functionCall", {})
                    args = func_call.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": f"call_{idx}_{hash(str(tc)) % 10000}",
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", ""),
                                "arguments": args,
                            },
                        }
                    )
                openai_message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            else:
                status = alt.get("status")
                finish_reason = "stop" if status == "ALTERNATIVE_STATUS_FINAL" else None

        usage_norm = {
            "prompt_tokens": int(usage.get("inputTextTokens", 0)),
            "completion_tokens": int(usage.get("completionTokens", 0)),
            "total_tokens": int(usage.get("totalTokens", 0)),
        }

        return ChatResult(
            message=openai_message,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage_norm,
            cost=None,
            raw=data,
        )


class GigaChatClient:
    def __init__(
        self,
        base_url: str,
        oauth_url: str,
        auth_key: str,
        model: str,
        scope: str = "GIGACHAT_API_PERS",
        verify_ssl: bool = True,
        timeout_seconds: int = 60,
        request_delay_seconds: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.oauth_url = oauth_url
        self.auth_key = auth_key
        self.model = model
        self.scope = scope
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = float(request_delay_seconds or 0.0)
        self._last_request_time: Optional[float] = None
        if not self.verify_ssl:
            urllib3.disable_warnings(InsecureRequestWarning)
        self.access_token: Optional[str] = None
        self.expires_at: float = 0.0

    def _throttle(self) -> None:
        if not self.request_delay_seconds or not self._last_request_time:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay_seconds:
            time.sleep(self.request_delay_seconds - elapsed)

    def _ensure_token(self) -> None:
        if self.access_token and time.time() < self.expires_at - 60:
            return
        headers = {
            "Authorization": f"Basic {self.auth_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        payload = {"scope": self.scope}
        self._throttle()
        resp = _request_with_retries(
            "POST",
            self.oauth_url,
            headers=headers,
            data=payload,
            verify=self.verify_ssl,
            timeout=self.timeout_seconds,
        )
        self._last_request_time = time.monotonic()
        try:
            data = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:200]
            raise RuntimeError(f"GigaChat auth error {resp.status_code}: non-JSON response: {snippet}")
        if resp.status_code >= 400:
            raise RuntimeError(f"GigaChat auth error {resp.status_code}: {data}")
        self.access_token = data.get("access_token")
        expires_in = data.get("expires_in", 300)
        self.expires_at = time.time() + float(expires_in)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        self._ensure_token()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if functions is not None:
            payload["functions"] = functions
            payload["function_call"] = "auto"

        self._throttle()
        resp = _request_with_retries(
            "POST",
            url,
            headers=headers,
            json=payload,
            verify=self.verify_ssl,
            timeout=self.timeout_seconds,
        )
        self._last_request_time = time.monotonic()
        try:
            data = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:200]
            raise RuntimeError(f"GigaChat HTTP {resp.status_code}: non-JSON response: {snippet}")
        if resp.status_code >= 400:
            raise RuntimeError(f"GigaChat HTTP {resp.status_code}: {data}")
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        tool_calls: List[Dict[str, Any]] = []
        if message.get("function_call"):
            fc = message["function_call"]
            tool_calls = [
                {
                    "id": f"gigachat_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": fc.get("name"), "arguments": fc.get("arguments")},
                }
            ]
        usage = data.get("usage") or {}
        cost = None
        for key in ("cost", "total_cost"):
            if key in data:
                try:
                    cost = float(data.get(key))
                except (TypeError, ValueError):
                    cost = None
                break
        if cost is None and isinstance(usage, dict) and "cost" in usage:
            try:
                cost = float(usage.get("cost"))
            except (TypeError, ValueError):
                cost = None
        return ChatResult(
            message=message,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            cost=cost,
            raw=data,
        )


def parse_tool_call_arguments(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function", {})
    args = function.get("arguments")
    return _parse_json_maybe(args)
