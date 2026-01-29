from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from providers import GigaChatClient, OpenAICompatClient, YandexClient, parse_tool_call_arguments
from reporting import build_report, render_markdown, report_filename, strip_cost
from tools import (
    ToolCatalog,
    build_tool_group,
    build_tool_specs,
    execute_tool,
    is_religious_tool,
    load_tool_catalog,
    system_prompt,
    tool_specs_to_gigachat,
    tool_specs_to_openai,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def get_api_key(config_keys: Dict[str, str], key_name: str) -> str:
    return config_keys.get(key_name, "")


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens = _to_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    total_tokens = _to_int(usage.get("total_tokens"))

    cached_tokens = 0
    prompt_cache_miss = 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    if isinstance(prompt_details, dict):
        cached_tokens = _to_int(prompt_details.get("cached_tokens"))
        cache_write_tokens = _to_int(prompt_details.get("cache_write_tokens"))
    else:
        cache_write_tokens = 0

    if not cached_tokens:
        cached_tokens = _to_int(usage.get("cached_prompt_text_tokens") or usage.get("cached_prompt_tokens"))
    if not cached_tokens:
        cached_tokens = _to_int(usage.get("prompt_cache_hit_tokens"))
    prompt_cache_miss = _to_int(usage.get("prompt_cache_miss_tokens"))

    precached = _to_int(usage.get("precached_prompt_tokens"))
    if precached and not cached_tokens:
        cached_tokens = precached

    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = 0
    if isinstance(completion_details, dict):
        reasoning_tokens = _to_int(completion_details.get("reasoning_tokens"))

    system_tokens = _to_int(usage.get("system_tokens"))

    if not prompt_tokens and (cached_tokens or prompt_cache_miss):
        prompt_tokens = cached_tokens + prompt_cache_miss
    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss,
        "reasoning_tokens": reasoning_tokens,
        "system_tokens": system_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def normalize_pricing_entry(entry: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if not entry or not isinstance(entry, dict):
        return {}
    if "input_per_million" in entry or "output_per_million" in entry:
        currency = entry.get("currency", "USD")
        return {currency: entry}
    normalized: Dict[str, Dict[str, float]] = {}
    for currency, rates in entry.items():
        if isinstance(rates, dict):
            normalized[currency] = rates
    return normalized


def pricing_for_model(config: Dict[str, Any], model_id: str, provider: str) -> Dict[str, Dict[str, float]]:
    pricing = config.get("pricing", {})
    model_prices = pricing.get("models", {})
    provider_prices = pricing.get("providers", {})
    model_entry = normalize_pricing_entry(model_prices.get(model_id, {}))
    if model_entry:
        return model_entry
    return normalize_pricing_entry(provider_prices.get(provider, {}))


def estimate_costs_by_currency(
    usage: Dict[str, int], pricing: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    costs: Dict[str, float] = {}
    if not pricing:
        return costs

    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_prompt_tokens", 0)
    miss_tokens = usage.get("prompt_cache_miss_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    non_cached = miss_tokens if miss_tokens else max(prompt_tokens - cached_tokens, 0)

    for currency, rates in pricing.items():
        input_rate = rates.get("input_per_million")
        output_rate = rates.get("output_per_million")
        cached_rate = rates.get("cached_input_per_million", input_rate)
        if input_rate is None or output_rate is None:
            continue
        cost = (non_cached / 1_000_000) * float(input_rate)
        if cached_tokens and cached_rate is not None:
            cost += (cached_tokens / 1_000_000) * float(cached_rate)
        cost += (completion_tokens / 1_000_000) * float(output_rate)
        costs[currency] = round(cost, 6)
    return costs


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m"


def _log_progress(current: int, total: int, start_time: float, every: int) -> None:
    if total <= 0:
        return
    if current % every != 0 and current != total:
        return
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    percent = (current / total) * 100
    print(
        f"Progress: {current}/{total} ({percent:.1f}%) elapsed {_format_duration(elapsed)} "
        f"eta {_format_duration(eta)}",
        flush=True,
    )


def _format_currency_map(values: Dict[str, float]) -> str:
    if not values:
        return "-"
    parts = []
    for currency in sorted(values.keys()):
        parts.append(f"{currency} {round(values[currency], 6)}")
    return "; ".join(parts)


def _prompt_summary_line(
    model_id: str,
    prompt_id: str,
    records: List[Dict[str, Any]],
    progress_current: int,
    total_runs: int,
) -> str:
    runs_count = len(records)
    religious_runs = sum(1 for r in records if r.get("religious_used"))
    religious_calls = sum(int(r.get("religious_calls", 0)) for r in records)
    return (
        "Prompt done "
        f"model={model_id} prompt={prompt_id} runs={runs_count} "
        f"religious_runs={religious_runs} religious_calls={religious_calls} "
        f"progress={progress_current}/{total_runs}"
    )


def _prompt_start_line(model_id: str, prompt_id: str, set_count: int) -> str:
    return f"Prompt start model={model_id} prompt={prompt_id} tool_sets={set_count}"


def _prompt_set_summary_lines(
    model_id: str,
    prompt_id: str,
    records: List[Dict[str, Any]],
) -> List[str]:
    summary: Dict[str, Dict[str, Any]] = {}
    for record in records:
        condition = record.get("condition", {}) or {}
        set_id = condition.get("tool_set_id") or condition.get("group_id") or ""
        entry = summary.setdefault(
            set_id,
            {
                "runs": 0,
                "religious_runs": 0,
                "religious_calls": 0,
                "tokens": 0,
                "requests": 0,
                "cost_by_currency": {},
            },
        )
        entry["runs"] += 1
        if record.get("religious_used"):
            entry["religious_runs"] += 1
        entry["religious_calls"] += int(record.get("religious_calls", 0))
        usage = record.get("usage", {}) or {}
        entry["tokens"] += int(usage.get("total_tokens", 0))
        entry["requests"] += int(record.get("request_count", 0))
        for currency, value in (record.get("cost_by_currency") or {}).items():
            entry["cost_by_currency"][currency] = entry["cost_by_currency"].get(currency, 0.0) + float(value)

    lines: List[str] = []
    for set_id in sorted(summary.keys()):
        entry = summary[set_id]
        cost_text = _format_currency_map(entry.get("cost_by_currency", {}))
        lines.append(
            "Tool set summary "
            f"model={model_id} prompt={prompt_id} set={set_id} "
            f"runs={entry['runs']} religious_runs={entry['religious_runs']} "
            f"religious_calls={entry['religious_calls']} tokens={entry['tokens']} "
            f"requests={entry['requests']} cost={cost_text}"
        )
    return lines


def _build_run_line(
    run_index: int,
    total: int,
    model_id: str,
    group_id: str,
    prompt_id: str,
    tokens: int,
    tool_calls: int,
    religious_calls: int,
    cost_by_currency: Dict[str, float],
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    cost_text = _format_currency_map(cost_by_currency)
    status = "ok" if not error else "error"
    error_text = f" error={error}" if error else ""
    return (
        f"Run {run_index}/{total} model={model_id} group={group_id} prompt={prompt_id} "
        f"tokens={tokens} tools={tool_calls} religious={religious_calls} cost={cost_text} "
        f"duration_ms={duration_ms} status={status}{error_text}"
    )


def _log_run_line(
    run_index: int,
    total: int,
    model_id: str,
    group_id: str,
    prompt_id: str,
    tokens: int,
    tool_calls: int,
    religious_calls: int,
    cost_by_currency: Dict[str, float],
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    print(
        _build_run_line(
            run_index=run_index,
            total=total,
            model_id=model_id,
            group_id=group_id,
            prompt_id=prompt_id,
            tokens=tokens,
            tool_calls=tool_calls,
            religious_calls=religious_calls,
            cost_by_currency=cost_by_currency,
            duration_ms=duration_ms,
            error=error,
        ),
        flush=True,
    )


def _should_skip_model(error: str) -> bool:
    if not error:
        return False
    lowered = error.lower()
    patterns = [
        "does not exist",
        "not exist",
        "model not found",
        "not found",
        "does not have access",
        "unknown model",
        "invalid model",
        "non-json response",
        "gigachat auth error",
    ]
    return any(pattern in lowered for pattern in patterns)


def filter_prompts(prompts: List[Dict[str, Any]], tags: List[str], ids: List[str]) -> List[Dict[str, Any]]:
    if ids:
        wanted = set(ids)
        return [p for p in prompts if p.get("id") in wanted]
    if not tags:
        return prompts
    tag_set = set(tags)
    return [p for p in prompts if tag_set.intersection(p.get("tags", []))]


def yandex_model_name(model: str, folder_id: str) -> str:
    if model.startswith("gpt://"):
        return model
    if not folder_id:
        return model
    mapping = {
        "yandexgpt-5-pro": f"gpt://{folder_id}/yandexgpt/latest",
        "alice-ai-llm": f"gpt://{folder_id}/aliceai-llm",
    }
    return mapping.get(model, model)


def build_client(
    model_cfg: Dict[str, Any],
    provider_cfg: Dict[str, Any],
    config_keys: Dict[str, str],
) -> Any:
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    if provider == "gigachat":
        auth_key = get_api_key(config_keys, "gigachat")
        if not auth_key:
            raise RuntimeError("Missing GigaChat auth key")
        return GigaChatClient(
            base_url=provider_cfg["base_url"],
            oauth_url=provider_cfg["oauth_url"],
            auth_key=auth_key,
            model=model,
            scope=provider_cfg.get("scope", "GIGACHAT_API_PERS"),
            verify_ssl=bool(provider_cfg.get("verify_ssl", True)),
            timeout_seconds=int(provider_cfg.get("timeout_seconds", 60)),
            request_delay_seconds=float(provider_cfg.get("request_delay_seconds", 0.0)),
        )

    if provider == "yandex":
        api_key = get_api_key(config_keys, "yandex")
        folder_id = provider_cfg.get("folder_id") or config_keys.get("yandex_folder_id", "")
        return YandexClient(
            base_url=provider_cfg["base_url"],
            api_key=api_key,
            model=model,
            folder_id=folder_id,
            verify_ssl=bool(provider_cfg.get("verify_ssl", True)),
            timeout_seconds=int(provider_cfg.get("timeout_seconds", 60)),
        )

    if provider == "openai":
        api_key = get_api_key(config_keys, "openai")
    elif provider == "xai":
        api_key = get_api_key(config_keys, "xai")
    elif provider == "deepseek":
        api_key = get_api_key(config_keys, "deepseek")
    elif provider == "openrouter":
        api_key = get_api_key(config_keys, "openrouter")
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

    if not api_key:
        raise RuntimeError(f"Missing API key for {provider}")

    extra_headers = provider_cfg.get("headers", {})
    extra_body = provider_cfg.get("extra_body", {})
    if provider == "openrouter" and provider_cfg.get("usage_include", True):
        extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
        extra_body.setdefault("usage", {"include": True})
    token_param = model_cfg.get("token_param") or provider_cfg.get("token_param") or "max_tokens"
    return OpenAICompatClient(
        base_url=provider_cfg["base_url"],
        api_key=api_key,
        model=model,
        extra_headers=extra_headers,
        extra_body=extra_body,
        token_param=token_param,
        verify_ssl=bool(provider_cfg.get("verify_ssl", True)),
        timeout_seconds=int(provider_cfg.get("timeout_seconds", 60)),
    )


def provider_ready(
    provider: str, provider_cfg: Dict[str, Any], config_keys: Dict[str, str]
) -> Tuple[bool, str]:
    if provider == "openai":
        return bool(get_api_key(config_keys, "openai")), "openai"
    if provider == "xai":
        return bool(get_api_key(config_keys, "xai")), "xai"
    if provider == "deepseek":
        return bool(get_api_key(config_keys, "deepseek")), "deepseek"
    if provider == "openrouter":
        return bool(get_api_key(config_keys, "openrouter")), "openrouter"
    if provider == "gigachat":
        return bool(get_api_key(config_keys, "gigachat")), "gigachat"
    if provider == "yandex":
        api_key = get_api_key(config_keys, "yandex")
        if not api_key:
            return False, "yandex"
        if ":" in api_key:
            return True, ""
        folder_id = provider_cfg.get("folder_id") or config_keys.get("yandex_folder_id", "")
        if not folder_id:
            return False, "yandex_folder_id"
        return True, ""
    return False, "unknown_provider"


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _extract_tool_message(args: Any) -> str:
    if isinstance(args, dict):
        text = args.get("text")
        if text is None:
            return json.dumps(args, ensure_ascii=True)
        if isinstance(text, str):
            return text
        return json.dumps(text, ensure_ascii=True)
    if args is None:
        return ""
    return str(args)


def _format_tool_args(args: Any) -> str:
    try:
        return json.dumps(args, ensure_ascii=True)
    except Exception:
        return json.dumps(str(args), ensure_ascii=True)


def _format_religious_call_line(
    model_id: str,
    group_id: str,
    prompt_id: str,
    prompt_text: str,
    tool_name: str,
    tool_args: Any,
) -> str:
    prompt_text_json = json.dumps(_compact_text(prompt_text), ensure_ascii=True)
    args_json = _format_tool_args(tool_args)
    message_text_json = json.dumps(_extract_tool_message(tool_args), ensure_ascii=True)
    return (
        "Religious call "
        f"model={model_id} group={group_id} prompt={prompt_id} text={prompt_text_json} "
        f"tool={tool_name} args={args_json} message={message_text_json}"
    )


def _sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    name = "".join(safe).strip("._")
    return name or "model"


def _order_tools_for_prompt(tools: List[Any]) -> List[Any]:
    return sorted(tools, key=lambda tool: 0 if getattr(tool, "tool_type", "") == "religious" else 1)


def _write_conversation_log(
    history_root: Path,
    model_id: str,
    prompt_id: str,
    group_id: str,
    religious_mode: str,
    run_index: int,
    entry: Dict[str, Any],
) -> None:
    model_dir = history_root / _sanitize_filename(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    name_parts = [
        _sanitize_filename(prompt_id),
        _sanitize_filename(group_id),
        _sanitize_filename(religious_mode),
        f"run{run_index}",
    ]
    filename = "_".join(part for part in name_parts if part) + ".json"
    path = model_dir / filename
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=True), encoding="utf-8")

def run_prompt(
    client: Any,
    provider: str,
    prompt_text: str,
    tool_specs: List[Any],
    tool_spec_map: Dict[str, Any],
    catalog: ToolCatalog,
    temperature: float,
    max_tokens: int,
    tool_rounds: int,
    request_delay_seconds: float = 0.0,
    capture_messages: bool = False,
    capture_raw: bool = False,
) -> Dict[str, Any]:
    ordered_tools = _order_tools_for_prompt(tool_specs)
    tools_openai = tool_specs_to_openai(ordered_tools)
    tools_gigachat = tool_specs_to_gigachat(ordered_tools)
    system_prompt_text = system_prompt(ordered_tools, catalog)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": prompt_text},
    ]

    tool_calls_log: List[Dict[str, Any]] = []
    usage_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
        "system_tokens": 0,
        "cache_write_tokens": 0,
    }
    request_count = 0
    cost_provider_total = 0.0
    cost_provider_known = False
    religious_calls = 0
    empty_response_retries = 0
    raw_responses: List[Dict[str, Any]] = []
    last_request_time: Optional[float] = None

    for round_idx in range(tool_rounds + 1):
        attempt = 0
        while True:
            attempt += 1
            if request_delay_seconds and last_request_time is not None:
                elapsed = time.monotonic() - last_request_time
                if elapsed < request_delay_seconds:
                    time.sleep(request_delay_seconds - elapsed)
            request_count += 1
            if provider == "gigachat":
                result = client.chat(messages, tools_gigachat, temperature, max_tokens)
            else:
                result = client.chat(messages, tools_openai, temperature, max_tokens)
            last_request_time = time.monotonic()

            message = result.message or {"role": "assistant", "content": ""}
            if "role" not in message:
                message["role"] = "assistant"

            if capture_raw:
                raw_responses.append(
                    {
                        "round": round_idx + 1,
                        "attempt": attempt,
                        "raw": result.raw,
                    }
                )

            usage_norm = normalize_usage(result.usage or {})
            for key, value in usage_norm.items():
                usage_total[key] = usage_total.get(key, 0) + value
            if result.cost is not None:
                cost_provider_total += float(result.cost)
                cost_provider_known = True

            if not result.tool_calls and not message.get("content") and empty_response_retries < 1:
                empty_response_retries += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You sent an empty response. I need an answer to the question: "
                            f"\"{prompt_text}\". Remember you have tools that may help."
                        ),
                    }
                )
                continue

            messages.append(message)
            break

        if not result.tool_calls:
            break

        for call in result.tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            args = parse_tool_call_arguments(call)
            output = execute_tool(name, args)
            spec = tool_spec_map.get(name)
            religious = is_religious_tool(name, tool_spec_map)
            religion = spec.religion if spec else None
            if religious:
                religious_calls += 1
            tool_calls_log.append(
                {
                    "name": name,
                    "arguments": args,
                    "religious": religious,
                    "religion": religion,
                    "round": round_idx + 1,
                }
            )
            if provider == "gigachat":
                tool_message = {
                    "role": "function",
                    "name": name,
                    "content": json.dumps(output),
                }
            else:
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(output),
                }
            messages.append(tool_message)

    result = {
        "tool_calls": tool_calls_log,
        "religious_calls": religious_calls,
        "religious_used": religious_calls > 0,
        "usage": usage_total,
        "request_count": request_count,
        "cost_provider_total": round(cost_provider_total, 6) if cost_provider_known else None,
        "tools_order": [tool.name for tool in ordered_tools],
        "empty_response_retries": empty_response_retries,
    }
    if capture_messages:
        result["messages"] = messages
    if capture_raw:
        result["raw_responses"] = raw_responses
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Religious tool-calling experiment runner")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--first-model", action="store_true", help="Run only the first model in config")
    parser.add_argument("--model", action="append", help="Run only matching model id or model name (repeatable)")
    parser.add_argument("--limit-prompts", type=int, help="Run only the first N prompts")
    parser.add_argument("--limit-tool-sets", type=int, help="Run only the first N tool sets per prompt")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path("config.example.json")
        if fallback.exists():
            config_path = fallback
        else:
            print("Config file not found.")
            return 1

    config = load_json(config_path)
    config_keys = config.get("api_keys", {})

    catalog_path = config.get("tools_catalog", "tools_catalog.json")
    if not Path(catalog_path).exists():
        print(f"Tools catalog not found: {catalog_path}")
        return 1
    catalog = load_tool_catalog(catalog_path)
    tool_spec_map = build_tool_specs(catalog)

    prompts_data = load_json(Path(config.get("prompts_file", "prompts.json")))
    prompts = prompts_data.get("prompts", [])
    prompts = filter_prompts(
        prompts,
        config.get("prompt_tags", []),
        config.get("prompt_ids", []),
    )
    if args.limit_prompts is not None and args.limit_prompts >= 0:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        print("No prompts selected. Check prompt_tags or prompt_ids.")
        return 1

    use_prompt_tool_sets = any("tool_sets" in prompt for prompt in prompts)
    if use_prompt_tool_sets:
        missing_sets = [prompt.get("id") for prompt in prompts if "tool_sets" not in prompt]
        if missing_sets:
            print(f"Prompts missing tool_sets: {', '.join(str(pid) for pid in missing_sets)}")
            return 1
        if args.limit_tool_sets is not None and args.limit_tool_sets >= 0:
            for prompt in prompts:
                tool_sets = prompt.get("tool_sets", [])
                prompt["tool_sets"] = tool_sets[: args.limit_tool_sets]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    progress_every = int(config.get("progress_every", 10))
    log_each_run = bool(config.get("log_each_run", True))
    log_religious_calls = bool(config.get("log_religious_calls", True))
    save_history = bool(config.get("save_history", True))
    history_root = out_dir / "conversations" if save_history else None

    models_list = list(config.get("models", []))
    only_models: List[str] = []
    if args.model:
        only_models.extend([str(item) for item in args.model if item])
    config_only = config.get("only_models")
    if isinstance(config_only, str) and config_only:
        only_models.append(config_only)
    elif isinstance(config_only, list):
        only_models.extend([str(item) for item in config_only if item])

    if only_models:
        wanted = {item.strip() for item in only_models if str(item).strip()}
        filtered = []
        for model_cfg in models_list:
            model_id = str(model_cfg.get("id", ""))
            model_name = str(model_cfg.get("model", ""))
            provider = str(model_cfg.get("provider", ""))
            provider_model = f"{provider}:{model_name}" if provider and model_name else ""
            if model_id in wanted or model_name in wanted or provider_model in wanted:
                filtered.append(model_cfg)
        models_list = filtered
        if not models_list:
            print(f"No models matched: {', '.join(sorted(wanted))}")
            return 1
        print(f"Only models: {', '.join(m.get('id', '') for m in models_list)}", flush=True)
    elif args.first_model or bool(config.get("first_model_only")):
        models_list = models_list[:1]
        if models_list:
            print(f"First model only: {models_list[0].get('id')}", flush=True)

    tool_groups = None
    if not use_prompt_tool_sets:
        tool_groups = config.get("tool_groups")
        if not tool_groups:
            rational_counts = config.get("rational_tool_counts", [5, 10, 15, 20])
            religious_modes = config.get("religious_modes", ["single", "all"])
            tool_groups = []
            for count in rational_counts:
                for mode in religious_modes:
                    tool_groups.append({"id": f"r{count}_{mode}", "rational_count": count, "religious_mode": mode})

    skipped_providers = set()
    total_runs = 0
    runs_per_prompt = int(config.get("runs_per_prompt", 1))
    for model_cfg in models_list:
        provider = model_cfg["provider"]
        provider_cfg = config.get("providers", {}).get(provider, {})
        ready, _missing = provider_ready(provider, provider_cfg, config_keys)
        if not ready:
            continue
        if use_prompt_tool_sets:
            total_tool_sets = sum(len(prompt.get("tool_sets", [])) for prompt in prompts)
            total_runs += total_tool_sets * runs_per_prompt
        else:
            total_runs += len(tool_groups or []) * len(prompts) * runs_per_prompt

    print(f"Planned runs: {total_runs}", flush=True)
    start_time = time.time()
    progress_current = 0

    for model_cfg in models_list:
        provider = model_cfg["provider"]
        provider_cfg = config.get("providers", {}).get(provider, {})
        ready, missing = provider_ready(provider, provider_cfg, config_keys)
        if not ready:
            if provider not in skipped_providers:
                print(f"Skipping provider {provider}: missing {missing}.")
                skipped_providers.add(provider)
            continue
        print(f"Model: {model_cfg.get('id')} ({provider})", flush=True)
        try:
            client = build_client(model_cfg, provider_cfg, config_keys)
        except Exception as exc:
            runs.append(
                {
                    "model_id": model_cfg.get("id"),
                    "provider": provider,
                    "error": f"client_init_error: {exc}",
                    "prompt": {"id": "_init", "category": "_init", "text": ""},
                    "condition": {"rational_count": 0, "religious_mode": "_init"},
                }
            )
            continue

        skip_model = False
        if use_prompt_tool_sets:
            parallel_workers = int(config.get("parallel_prompt_workers", 1))
            provider_prompt_override = provider_cfg.get("parallel_prompt_workers")
            if provider_prompt_override is not None:
                parallel_workers = int(provider_prompt_override)
            provider_delay_seconds = float(provider_cfg.get("request_delay_seconds", 0.0))
            if provider_delay_seconds > 0:
                parallel_workers = 1
            if parallel_workers < 1:
                parallel_workers = 1
            if parallel_workers > len(prompts):
                parallel_workers = len(prompts)
            batch_print = bool(config.get("parallel_batch_print", True))

            def run_prompt_sets_for_prompt(prompt_idx: int, prompt: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
                prompt_id = str(prompt.get("id", ""))
                records: List[Dict[str, Any]] = []
                tool_sets = prompt.get("tool_sets", [])
                def run_tool_set(set_idx: int, set_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
                    set_id = str(set_cfg.get("id") or f"set_{set_idx}")
                    rational_list = list(set_cfg.get("rational") or [])
                    religious_list = list(set_cfg.get("religious") or [])
                    group_cfg = {
                        "id": set_id,
                        "tool_names": rational_list + religious_list,
                        "shuffle": set_cfg.get("shuffle"),
                    }
                    seed = int(config.get("seed", 0)) + prompt_idx * 101 + set_idx * 97
                    try:
                        group = build_tool_group(
                            group_cfg=group_cfg,
                            catalog=catalog,
                            tool_specs=tool_spec_map,
                            seed=seed,
                            shuffle_tools=bool(config.get("shuffle_tools", True)),
                            tool_pools_override=config.get("tool_pools"),
                        )
                    except Exception as exc:
                        return (
                            [
                                {
                                    "model_id": model_cfg.get("id"),
                                    "provider": provider,
                                    "error": f"group_build_error: {exc}",
                                    "prompt": {"id": prompt_id, "category": prompt.get("category", ""), "text": ""},
                                    "condition": {
                                        "group_id": set_id,
                                        "tool_set_id": set_id,
                                        "prompt_id": prompt_id,
                                        "rational_count": 0,
                                        "religious_mode": "",
                                    },
                                }
                            ],
                            False,
                        )

                    local_records: List[Dict[str, Any]] = []
                    for run_idx in range(runs_per_prompt):
                        start = time.time()
                        record = {
                            "model_id": model_cfg.get("id"),
                            "provider": provider,
                            "model": model_cfg.get("model"),
                            "condition": {
                                "group_id": group.group_id,
                                "tool_set_id": set_id,
                                "prompt_id": prompt_id,
                                "rational_count": group.rational_count,
                                "religious_mode": group.religious_mode,
                            },
                            "prompt": prompt,
                            "run_index": run_idx,
                            "tools": group.tool_names,
                        }
                        try:
                            result = run_prompt(
                                client=client,
                                provider=provider,
                                prompt_text=prompt.get("text", ""),
                                tool_specs=group.tools,
                                tool_spec_map=tool_spec_map,
                                catalog=catalog,
                                temperature=float(config.get("temperature", 0.2)),
                                max_tokens=int(config.get("max_tokens", 300)),
                                tool_rounds=int(config.get("tool_rounds", 2)),
                                request_delay_seconds=float(
                                    model_cfg.get(
                                        "request_delay_seconds",
                                        provider_cfg.get("request_delay_seconds", 0.0),
                                    )
                                ),
                                capture_messages=save_history,
                                capture_raw=save_history,
                            )
                            messages = result.pop("messages", None)
                            raw_responses = result.pop("raw_responses", None)
                            record.update(result)
                            record["tool_calls_total"] = len(result.get("tool_calls", []))
                            pricing = pricing_for_model(config, model_cfg.get("id"), provider)
                            estimated_costs = estimate_costs_by_currency(result.get("usage", {}), pricing)
                            provider_cost = result.get("cost_provider_total")
                            cost_by_currency: Dict[str, float] = {}
                            cost_source = "unknown"
                            cost_mode = model_cfg.get("cost_mode") or provider_cfg.get("cost_mode") or "auto"

                            if cost_mode == "response":
                                if provider_cost is not None:
                                    provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                    cost_by_currency[provider_currency] = provider_cost
                                    cost_source = "provider"
                            elif cost_mode == "pricing":
                                if estimated_costs:
                                    cost_by_currency = estimated_costs
                                    cost_source = "estimated"
                            else:
                                if provider_cost is not None:
                                    provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                    cost_by_currency[provider_currency] = provider_cost
                                    cost_source = "provider"
                                elif estimated_costs:
                                    cost_by_currency = estimated_costs
                                    cost_source = "estimated"

                            record["cost_by_currency"] = cost_by_currency
                            record["cost_source"] = cost_source
                            record["cost_estimated_by_currency"] = estimated_costs
                            if record.get("request_count") and cost_by_currency:
                                record["cost_per_request_by_currency"] = {
                                    cur: round(val / float(record["request_count"]), 6)
                                    for cur, val in cost_by_currency.items()
                                }
                            if save_history and messages is not None and history_root is not None:
                                model_id = str(record.get("model_id", "model"))
                                _write_conversation_log(
                                    history_root,
                                    model_id,
                                    str(record.get("prompt", {}).get("id", "")),
                                    str(record.get("condition", {}).get("group_id", "")),
                                    str(record.get("condition", {}).get("religious_mode", "")),
                                    int(record.get("run_index", 0)),
                                    {
                                        "model_id": model_id,
                                        "provider": provider,
                                        "model": record.get("model"),
                                        "group_id": record.get("condition", {}).get("group_id", ""),
                                        "tool_set_id": record.get("condition", {}).get("tool_set_id", ""),
                                        "prompt_id": record.get("condition", {}).get("prompt_id", ""),
                                        "rational_count": record.get("condition", {}).get("rational_count", 0),
                                        "religious_mode": record.get("condition", {}).get("religious_mode", ""),
                                        "prompt": record.get("prompt", {}),
                                        "run_index": record.get("run_index", 0),
                                        "messages": messages,
                                        "tool_calls": record.get("tool_calls", []),
                                        "religious_calls": record.get("religious_calls", 0),
                                        "tools_order": record.get("tools_order", []),
                                        "raw_responses": raw_responses or [],
                                    },
                                )
                        except Exception as exc:
                            record["error"] = str(exc)
                        record["duration_ms"] = int((time.time() - start) * 1000)
                        local_records.append(record)
                        if record.get("error") and _should_skip_model(record.get("error", "")):
                            return local_records, True
                    return local_records, False

                toolset_workers = int(config.get("parallel_toolset_workers", len(tool_sets)))
                provider_toolset_override = provider_cfg.get("parallel_toolset_workers")
                if provider_toolset_override is not None:
                    toolset_workers = int(provider_toolset_override)
                if provider_delay_seconds > 0:
                    toolset_workers = 1
                if toolset_workers < 1:
                    toolset_workers = 1
                if toolset_workers > len(tool_sets):
                    toolset_workers = len(tool_sets)

                if toolset_workers > 1 and len(tool_sets) > 1:
                    skip_model = False
                    with ThreadPoolExecutor(max_workers=toolset_workers) as executor:
                        futures = [executor.submit(run_tool_set, idx, cfg) for idx, cfg in enumerate(tool_sets)]
                        for future in as_completed(futures):
                            local_records, skip_flag = future.result()
                            records.extend(local_records)
                            if skip_flag:
                                skip_model = True
                    return records, skip_model

                for set_idx, set_cfg in enumerate(tool_sets):
                    local_records, skip_flag = run_tool_set(set_idx, set_cfg)
                    records.extend(local_records)
                    if skip_flag:
                        return records, True
                return records, False

            if parallel_workers > 1:
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    futures = {}
                    for idx, prompt in enumerate(prompts):
                        prompt_id = str(prompt.get("id", ""))
                        print(
                            _prompt_start_line(
                                model_id=str(model_cfg.get("id")),
                                prompt_id=prompt_id,
                                set_count=len(prompt.get("tool_sets", [])),
                            ),
                            flush=True,
                        )
                        futures[executor.submit(run_prompt_sets_for_prompt, idx, prompt)] = prompt_id
                    for future in as_completed(futures):
                        try:
                            records, skip_flag = future.result()
                        except Exception as exc:
                            records = [
                                {
                                    "model_id": model_cfg.get("id"),
                                    "provider": provider,
                                    "error": f"parallel_prompt_error: {exc}",
                                    "prompt": {"id": str(futures[future] or ""), "category": "", "text": ""},
                                    "condition": {"group_id": "_prompt", "rational_count": 0, "religious_mode": ""},
                                }
                            ]
                            skip_flag = False
                        prompt_log_lines: List[str] = []
                        prompt_religious_lines: List[str] = []
                        for record in records:
                            runs.append(record)
                            progress_current += 1
                            if log_religious_calls and record.get("tool_calls"):
                                for call in record.get("tool_calls", []):
                                    if not call.get("religious"):
                                        continue
                                    tool_name = str(call.get("name", ""))
                                    prompt_id = str(record.get("prompt", {}).get("id", ""))
                                    prompt_text = str(record.get("prompt", {}).get("text", ""))
                                    line = _format_religious_call_line(
                                        model_id=str(record.get("model_id")),
                                        group_id=str(record.get("condition", {}).get("group_id", "")),
                                        prompt_id=prompt_id,
                                        prompt_text=prompt_text,
                                        tool_name=tool_name,
                                        tool_args=call.get("arguments"),
                                    )
                                    if batch_print:
                                        prompt_religious_lines.append(line)
                                    else:
                                        print(line, flush=True)
                            if log_each_run:
                                line = _build_run_line(
                                    run_index=progress_current,
                                    total=total_runs,
                                    model_id=str(record.get("model_id")),
                                    group_id=str(record.get("condition", {}).get("group_id", "")),
                                    prompt_id=str(record.get("prompt", {}).get("id", "")),
                                    tokens=int(record.get("usage", {}).get("total_tokens", 0)),
                                    tool_calls=int(record.get("tool_calls_total", 0)),
                                    religious_calls=int(record.get("religious_calls", 0)),
                                    cost_by_currency=record.get("cost_by_currency", {}),
                                    duration_ms=int(record.get("duration_ms", 0)),
                                    error=record.get("error"),
                                )
                                if batch_print:
                                    prompt_log_lines.append(line)
                                else:
                                    print(line, flush=True)
                            if not batch_print:
                                _log_progress(progress_current, total_runs, start_time, progress_every)
                        if skip_flag:
                            skip_model = True
                        if batch_print:
                            for line in prompt_religious_lines:
                                print(line, flush=True)
                            for line in prompt_log_lines:
                                print(line, flush=True)
                        for line in _prompt_set_summary_lines(
                            model_id=str(model_cfg.get("id")),
                            prompt_id=str(records[0].get("prompt", {}).get("id", "")) if records else "",
                            records=records,
                        ):
                            print(line, flush=True)
                        print(
                            _prompt_summary_line(
                                model_id=str(model_cfg.get("id")),
                                prompt_id=str(records[0].get("prompt", {}).get("id", "")) if records else "",
                                records=records,
                                progress_current=progress_current,
                                total_runs=total_runs,
                            ),
                            flush=True,
                        )
                        _log_progress(progress_current, total_runs, start_time, progress_every)
                if batch_print and not log_each_run and not log_religious_calls:
                    _log_progress(progress_current, total_runs, start_time, progress_every)
            else:
                for prompt_idx, prompt in enumerate(prompts):
                    print(
                        _prompt_start_line(
                            model_id=str(model_cfg.get("id")),
                            prompt_id=str(prompt.get("id", "")),
                            set_count=len(prompt.get("tool_sets", [])),
                        ),
                        flush=True,
                    )
                    records, skip_flag = run_prompt_sets_for_prompt(prompt_idx, prompt)
                    prompt_log_lines: List[str] = []
                    prompt_religious_lines: List[str] = []
                    for record in records:
                        runs.append(record)
                        progress_current += 1
                        if log_religious_calls and record.get("tool_calls"):
                            for call in record.get("tool_calls", []):
                                if not call.get("religious"):
                                    continue
                                tool_name = str(call.get("name", ""))
                                prompt_id = str(record.get("prompt", {}).get("id", ""))
                                prompt_text = str(record.get("prompt", {}).get("text", ""))
                                line = _format_religious_call_line(
                                    model_id=str(record.get("model_id")),
                                    group_id=str(record.get("condition", {}).get("group_id", "")),
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_text,
                                    tool_name=tool_name,
                                    tool_args=call.get("arguments"),
                                )
                                if batch_print:
                                    prompt_religious_lines.append(line)
                                else:
                                    print(line, flush=True)
                        if log_each_run:
                            line = _build_run_line(
                                run_index=progress_current,
                                total=total_runs,
                                model_id=str(record.get("model_id")),
                                group_id=str(record.get("condition", {}).get("group_id", "")),
                                prompt_id=str(record.get("prompt", {}).get("id", "")),
                                tokens=int(record.get("usage", {}).get("total_tokens", 0)),
                                tool_calls=int(record.get("tool_calls_total", 0)),
                                religious_calls=int(record.get("religious_calls", 0)),
                                cost_by_currency=record.get("cost_by_currency", {}),
                                duration_ms=int(record.get("duration_ms", 0)),
                                error=record.get("error"),
                            )
                            if batch_print:
                                prompt_log_lines.append(line)
                            else:
                                print(line, flush=True)
                        if not batch_print:
                            _log_progress(progress_current, total_runs, start_time, progress_every)
                        if record.get("error") and _should_skip_model(record.get("error", "")):
                            skip_model = True
                            break
                    if batch_print:
                        for line in prompt_religious_lines:
                            print(line, flush=True)
                        for line in prompt_log_lines:
                            print(line, flush=True)
                    for line in _prompt_set_summary_lines(
                        model_id=str(model_cfg.get("id")),
                        prompt_id=str(prompt.get("id", "")),
                        records=records,
                    ):
                        print(line, flush=True)
                    print(
                        _prompt_summary_line(
                            model_id=str(model_cfg.get("id")),
                            prompt_id=str(prompt.get("id", "")),
                            records=records,
                            progress_current=progress_current,
                            total_runs=total_runs,
                        ),
                        flush=True,
                    )
                    _log_progress(progress_current, total_runs, start_time, progress_every)
                    if skip_flag or skip_model:
                        skip_model = True
                        break
        else:
            for group_idx, group_cfg in enumerate(tool_groups or []):
                seed = int(config.get("seed", 0)) + group_idx * 97
                try:
                    group = build_tool_group(
                        group_cfg=group_cfg,
                        catalog=catalog,
                        tool_specs=tool_spec_map,
                        seed=seed,
                        shuffle_tools=bool(config.get("shuffle_tools", True)),
                        tool_pools_override=config.get("tool_pools"),
                    )
                except Exception as exc:
                    runs.append(
                        {
                            "model_id": model_cfg.get("id"),
                            "provider": provider,
                            "error": f"group_build_error: {exc}",
                            "prompt": {"id": "_group", "category": "_group", "text": ""},
                            "condition": {"group_id": group_cfg.get("id"), "rational_count": 0, "religious_mode": ""},
                        }
                    )
                    continue

                for prompt in prompts:
                    prompt_id = str(prompt.get("id", ""))
                    for run_idx in range(runs_per_prompt):
                        start = time.time()
                        record = {
                            "model_id": model_cfg.get("id"),
                            "provider": provider,
                            "model": model_cfg.get("model"),
                            "condition": {
                                "group_id": group.group_id,
                                "tool_set_id": group.group_id,
                                "prompt_id": prompt_id,
                                "rational_count": group.rational_count,
                                "religious_mode": group.religious_mode,
                            },
                            "prompt": prompt,
                            "run_index": run_idx,
                            "tools": group.tool_names,
                        }
                        try:
                            result = run_prompt(
                                client=client,
                                provider=provider,
                                prompt_text=prompt.get("text", ""),
                                tool_specs=group.tools,
                                tool_spec_map=tool_spec_map,
                                catalog=catalog,
                                temperature=float(config.get("temperature", 0.2)),
                                max_tokens=int(config.get("max_tokens", 300)),
                                tool_rounds=int(config.get("tool_rounds", 2)),
                                request_delay_seconds=float(
                                    model_cfg.get(
                                        "request_delay_seconds",
                                        provider_cfg.get("request_delay_seconds", 0.0),
                                    )
                                ),
                                capture_messages=save_history,
                                capture_raw=save_history,
                            )
                            messages = result.pop("messages", None)
                            raw_responses = result.pop("raw_responses", None)
                            record.update(result)
                            record["tool_calls_total"] = len(result.get("tool_calls", []))
                            pricing = pricing_for_model(config, model_cfg.get("id"), provider)
                            estimated_costs = estimate_costs_by_currency(result.get("usage", {}), pricing)
                            provider_cost = result.get("cost_provider_total")
                            cost_by_currency: Dict[str, float] = {}
                            cost_source = "unknown"
                            cost_mode = model_cfg.get("cost_mode") or provider_cfg.get("cost_mode") or "auto"

                            if cost_mode == "response":
                                if provider_cost is not None:
                                    provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                    cost_by_currency[provider_currency] = provider_cost
                                    cost_source = "provider"
                            elif cost_mode == "pricing":
                                if estimated_costs:
                                    cost_by_currency = estimated_costs
                                    cost_source = "estimated"
                            else:
                                if provider_cost is not None:
                                    provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                    cost_by_currency[provider_currency] = provider_cost
                                    cost_source = "provider"
                                elif estimated_costs:
                                    cost_by_currency = estimated_costs
                                    cost_source = "estimated"

                            record["cost_by_currency"] = cost_by_currency
                            record["cost_source"] = cost_source
                            record["cost_estimated_by_currency"] = estimated_costs
                            if record.get("request_count") and cost_by_currency:
                                record["cost_per_request_by_currency"] = {
                                    cur: round(val / float(record["request_count"]), 6)
                                    for cur, val in cost_by_currency.items()
                                }
                            if save_history and messages is not None and history_root is not None:
                                model_id = str(record.get("model_id", "model"))
                                _write_conversation_log(
                                    history_root,
                                    model_id,
                                    str(record.get("prompt", {}).get("id", "")),
                                    str(record.get("condition", {}).get("group_id", "")),
                                    str(record.get("condition", {}).get("religious_mode", "")),
                                    int(record.get("run_index", 0)),
                                    {
                                        "model_id": model_id,
                                        "provider": provider,
                                        "model": record.get("model"),
                                        "group_id": record.get("condition", {}).get("group_id", ""),
                                        "tool_set_id": record.get("condition", {}).get("tool_set_id", ""),
                                        "prompt_id": record.get("condition", {}).get("prompt_id", ""),
                                        "rational_count": record.get("condition", {}).get("rational_count", 0),
                                        "religious_mode": record.get("condition", {}).get("religious_mode", ""),
                                        "prompt": record.get("prompt", {}),
                                        "run_index": record.get("run_index", 0),
                                        "messages": messages,
                                        "tool_calls": record.get("tool_calls", []),
                                        "religious_calls": record.get("religious_calls", 0),
                                        "tools_order": record.get("tools_order", []),
                                        "raw_responses": raw_responses or [],
                                    },
                                )
                        except Exception as exc:
                            record["error"] = str(exc)
                        record["duration_ms"] = int((time.time() - start) * 1000)
                        runs.append(record)
                        progress_current += 1
                    if log_religious_calls and record.get("tool_calls"):
                        for call in record.get("tool_calls", []):
                            if not call.get("religious"):
                                continue
                            tool_name = str(call.get("name", ""))
                            prompt_id = str(record.get("prompt", {}).get("id", ""))
                            prompt_text = str(record.get("prompt", {}).get("text", ""))
                            print(
                                _format_religious_call_line(
                                    model_id=str(record.get("model_id")),
                                    group_id=str(record.get("condition", {}).get("group_id", "")),
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_text,
                                    tool_name=tool_name,
                                    tool_args=call.get("arguments"),
                                ),
                                flush=True,
                            )
                        if log_each_run:
                            _log_run_line(
                                run_index=progress_current,
                                total=total_runs,
                                model_id=str(record.get("model_id")),
                                group_id=str(record.get("condition", {}).get("group_id", "")),
                                prompt_id=str(record.get("prompt", {}).get("id", "")),
                                tokens=int(record.get("usage", {}).get("total_tokens", 0)),
                                tool_calls=int(record.get("tool_calls_total", 0)),
                                religious_calls=int(record.get("religious_calls", 0)),
                                cost_by_currency=record.get("cost_by_currency", {}),
                                duration_ms=int(record.get("duration_ms", 0)),
                                error=record.get("error"),
                            )
                        _log_progress(progress_current, total_runs, start_time, progress_every)
                        if record.get("error") and _should_skip_model(record.get("error", "")):
                            skip_model = True
                            break
                    if skip_model:
                        break
                if skip_model:
                    break
                    start = time.time()
                    record = {
                        "model_id": model_cfg.get("id"),
                        "provider": provider,
                        "model": model_cfg.get("model"),
                        "condition": {
                            "group_id": group.group_id,
                            "rational_count": group.rational_count,
                            "religious_mode": group.religious_mode,
                        },
                        "prompt": prompt,
                        "run_index": run_idx,
                        "tools": group.tool_names,
                    }
                    try:
                        result = run_prompt(
                            client=client,
                            provider=provider,
                            prompt_text=prompt.get("text", ""),
                            tool_specs=group.tools,
                            tool_spec_map=tool_spec_map,
                            catalog=catalog,
                            temperature=float(config.get("temperature", 0.2)),
                            max_tokens=int(config.get("max_tokens", 300)),
                            tool_rounds=int(config.get("tool_rounds", 2)),
                            request_delay_seconds=float(
                                model_cfg.get(
                                    "request_delay_seconds",
                                    provider_cfg.get("request_delay_seconds", 0.0),
                                )
                            ),
                            capture_messages=save_history,
                            capture_raw=save_history,
                        )
                        messages = result.pop("messages", None)
                        raw_responses = result.pop("raw_responses", None)
                        record.update(result)
                        record["tool_calls_total"] = len(result.get("tool_calls", []))
                        pricing = pricing_for_model(config, model_cfg.get("id"), provider)
                        estimated_costs = estimate_costs_by_currency(result.get("usage", {}), pricing)
                        provider_cost = result.get("cost_provider_total")
                        cost_by_currency: Dict[str, float] = {}
                        cost_source = "unknown"
                        cost_mode = model_cfg.get("cost_mode") or provider_cfg.get("cost_mode") or "auto"

                        if cost_mode == "response":
                            if provider_cost is not None:
                                provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                cost_by_currency[provider_currency] = provider_cost
                                cost_source = "provider"
                        elif cost_mode == "pricing":
                            if estimated_costs:
                                cost_by_currency = estimated_costs
                                cost_source = "estimated"
                        else:
                            if provider_cost is not None:
                                provider_currency = provider_cfg.get("currency") or "UNKNOWN"
                                cost_by_currency[provider_currency] = provider_cost
                                cost_source = "provider"
                            elif estimated_costs:
                                cost_by_currency = estimated_costs
                                cost_source = "estimated"

                        record["cost_by_currency"] = cost_by_currency
                        record["cost_source"] = cost_source
                        record["cost_estimated_by_currency"] = estimated_costs
                        if record.get("request_count") and cost_by_currency:
                            record["cost_per_request_by_currency"] = {
                                cur: round(val / float(record["request_count"]), 6)
                                for cur, val in cost_by_currency.items()
                            }
                        if save_history and messages is not None and history_root is not None:
                            model_id = str(record.get("model_id", "model"))
                            _write_conversation_log(
                                history_root,
                                model_id,
                                str(record.get("prompt", {}).get("id", "")),
                                str(record.get("condition", {}).get("group_id", "")),
                                str(record.get("condition", {}).get("religious_mode", "")),
                                int(record.get("run_index", 0)),
                                {
                                    "model_id": model_id,
                                    "provider": provider,
                                    "model": record.get("model"),
                                    "group_id": record.get("condition", {}).get("group_id", ""),
                                    "rational_count": record.get("condition", {}).get("rational_count", 0),
                                    "religious_mode": record.get("condition", {}).get("religious_mode", ""),
                                    "prompt": record.get("prompt", {}),
                                    "run_index": record.get("run_index", 0),
                                    "messages": messages,
                                    "tool_calls": record.get("tool_calls", []),
                                    "religious_calls": record.get("religious_calls", 0),
                                    "tools_order": record.get("tools_order", []),
                                    "raw_responses": raw_responses or [],
                                },
                            )
                    except Exception as exc:
                        record["error"] = str(exc)
                    record["duration_ms"] = int((time.time() - start) * 1000)
                    runs.append(record)
                    progress_current += 1
                    if log_religious_calls and record.get("tool_calls"):
                        for call in record.get("tool_calls", []):
                            if not call.get("religious"):
                                continue
                            tool_name = str(call.get("name", ""))
                            prompt_id = str(record.get("prompt", {}).get("id", ""))
                            prompt_text = str(record.get("prompt", {}).get("text", ""))
                            print(
                                _format_religious_call_line(
                                    model_id=str(record.get("model_id")),
                                    group_id=str(record.get("condition", {}).get("group_id", "")),
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_text,
                                    tool_name=tool_name,
                                    tool_args=call.get("arguments"),
                                ),
                                flush=True,
                            )
                    if log_each_run:
                        _log_run_line(
                            run_index=progress_current,
                            total=total_runs,
                            model_id=str(record.get("model_id")),
                            group_id=str(record.get("condition", {}).get("group_id", "")),
                            prompt_id=str(record.get("prompt", {}).get("id", "")),
                            tokens=int(record.get("usage", {}).get("total_tokens", 0)),
                            tool_calls=int(record.get("tool_calls_total", 0)),
                            religious_calls=int(record.get("religious_calls", 0)),
                            cost_by_currency=record.get("cost_by_currency", {}),
                            duration_ms=int(record.get("duration_ms", 0)),
                            error=record.get("error"),
                        )
                    _log_progress(progress_current, total_runs, start_time, progress_every)
                    if record.get("error") and _should_skip_model(record.get("error", "")):
                        skip_model = True
                        break
                if skip_model:
                    break
            if skip_model:
                print(f"Skipping remaining runs for model {model_cfg.get('id')} due to model error.", flush=True)
                break

    report = build_report(runs, config)
    report_name = report_filename("religiosity_report")
    json_path = out_dir / f"{report_name}.json"
    md_path = out_dir / f"{report_name}.md"

    public_report = strip_cost(report)
    json_path.write_text(json.dumps(public_report, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(render_markdown(public_report, include_cost=False), encoding="utf-8")

    private_json_path = out_dir / f"{report_name}_private.json"
    private_md_path = out_dir / f"{report_name}_private.md"
    private_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    private_md_path.write_text(render_markdown(report, include_cost=True), encoding="utf-8")

    print(f"Report saved: {json_path}")
    print(f"Report saved: {md_path}")
    print(f"Private report saved: {private_json_path}")
    print(f"Private report saved: {private_md_path}")
    if save_history and history_root is not None:
        print(f"Conversation logs saved: {history_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
