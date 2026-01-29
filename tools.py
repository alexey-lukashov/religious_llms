from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for very old Python
    ZoneInfo = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    tool_type: str
    religion: Optional[str] = None


@dataclass
class ToolCatalog:
    tools: Dict[str, Dict[str, Any]]
    rational_pool: List[str]
    religious_pool: List[str]
    system_prompt_template: str
    tool_line_template: str


@dataclass
class ToolGroupResult:
    group_id: str
    tools: List[ToolSpec]
    tool_names: List[str]
    rational_count: int
    religious_count: int
    religious_mode: str


# -------------------------
# Rational tool handlers
# -------------------------


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
    )

    for subnode in ast.walk(node):
        if not isinstance(subnode, allowed_nodes):
            raise ValueError("Unsupported expression")

    return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}})


def calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(args.get("expression", ""))
    try:
        value = _safe_eval(expr)
        return {"ok": True, "result": value}
    except Exception as exc:
        return {"ok": False, "error": f"calculator_error: {exc}"}


def date_diff(args: Dict[str, Any]) -> Dict[str, Any]:
    date1 = args.get("date1")
    date2 = args.get("date2")
    try:
        d1 = datetime.fromisoformat(date1)
        d2 = datetime.fromisoformat(date2)
        return {"ok": True, "days": (d2 - d1).days}
    except Exception as exc:
        return {"ok": False, "error": f"date_diff_error: {exc}"}


def route_train_left(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "route": "left"}


def route_train_right(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "route": "right"}


def random_number(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        low = int(args.get("min", 0))
        high = int(args.get("max", 100))
        if low > high:
            low, high = high, low
        return {"ok": True, "value": random.randint(low, high)}
    except Exception as exc:
        return {"ok": False, "error": f"random_number_error: {exc}"}


def sort_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    nums = args.get("numbers", [])
    try:
        return {"ok": True, "sorted": sorted(float(n) for n in nums)}
    except Exception as exc:
        return {"ok": False, "error": f"sort_numbers_error: {exc}"}


def word_count(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return {"ok": True, "words": len(words)}


def make_uuid(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "uuid": str(uuid.uuid4())}


def summarize_text(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    if not text:
        return {"ok": False, "error": "summarize_text_error: empty text"}
    summary = text.strip().split(".")
    summary = ".".join(summary[:2]).strip()
    if summary and not summary.endswith("."):
        summary += "."
    if not summary:
        summary = text[:120].strip()
    return {"ok": True, "summary": summary}


def translate_text(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    lang = str(args.get("target_lang", ""))
    if not lang:
        return {"ok": False, "error": "translate_text_error: target_lang required"}
    return {"ok": True, "translation": f"[translated to {lang}] {text}"}


def define_word(args: Dict[str, Any]) -> Dict[str, Any]:
    word = str(args.get("word", "")).lower()
    dictionary = {
        "religion": "a system of beliefs, practices, and worship.",
        "science": "the systematic study of the structure and behavior of the physical world.",
        "ethics": "moral principles that govern behavior.",
    }
    meaning = dictionary.get(word)
    if not meaning:
        return {"ok": False, "error": "define_word_error: word not found"}
    return {"ok": True, "definition": meaning}


def convert_units(args: Dict[str, Any]) -> Dict[str, Any]:
    value = args.get("value")
    from_unit = str(args.get("from_unit", "")).lower()
    to_unit = str(args.get("to_unit", "")).lower()
    try:
        value = float(value)
    except Exception:
        return {"ok": False, "error": "convert_units_error: invalid value"}

    conversions = {
        ("cm", "in"): lambda v: v / 2.54,
        ("in", "cm"): lambda v: v * 2.54,
        ("kg", "lb"): lambda v: v * 2.2046226218,
        ("lb", "kg"): lambda v: v / 2.2046226218,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
    }

    func = conversions.get((from_unit, to_unit))
    if not func:
        return {"ok": False, "error": "convert_units_error: unsupported units"}
    return {"ok": True, "value": func(value), "unit": to_unit}


def timezone_time(args: Dict[str, Any]) -> Dict[str, Any]:
    tz = str(args.get("timezone", ""))
    if not tz:
        return {"ok": False, "error": "timezone_time_error: timezone required"}
    try:
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo unavailable")
        now = datetime.now(ZoneInfo(tz))
        return {"ok": True, "iso": now.isoformat()}
    except Exception as exc:
        return {"ok": False, "error": f"timezone_time_error: {exc}"}


def extract_keywords(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    tokens = [t.strip(".,!?;:\"'()[]{}").lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return {"ok": True, "keywords": unique[:8]}


def sentiment(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", "")).lower()
    positive = ["good", "great", "happy", "love", "excellent", "wonderful"]
    negative = ["bad", "sad", "angry", "hate", "terrible", "awful"]
    score = sum(1 for w in positive if w in text) - sum(1 for w in negative if w in text)
    label = "neutral"
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    return {"ok": True, "sentiment": label, "score": score}


def sha256_hash(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"ok": True, "sha256": digest}


def base64_encode(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {"ok": True, "base64": encoded}


def base64_decode(args: Dict[str, Any]) -> Dict[str, Any]:
    data = str(args.get("base64", ""))
    try:
        decoded = base64.b64decode(data, validate=True)
    except Exception as exc:
        return {"ok": False, "error": f"base64_decode_error: {exc}"}
    try:
        text = decoded.decode("utf-8")
    except Exception:
        text = decoded.decode("utf-8", errors="replace")
    return {"ok": True, "text": text}


def csv_column_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    csv_text = str(args.get("csv_text", ""))
    column = str(args.get("column", ""))
    if not csv_text or not column:
        return {"ok": False, "error": "csv_column_stats_error: csv_text and column required"}
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames or column not in reader.fieldnames:
        return {"ok": False, "error": "csv_column_stats_error: column not found"}
    values: List[float] = []
    for row in reader:
        raw = row.get(column, "")
        try:
            values.append(float(raw))
        except Exception:
            continue
    if not values:
        return {"ok": False, "error": "csv_column_stats_error: no numeric values"}
    count = len(values)
    total = sum(values)
    return {
        "ok": True,
        "count": count,
        "min": min(values),
        "max": max(values),
        "mean": round(total / count, 6),
    }


def regex_find(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = str(args.get("pattern", ""))
    text = str(args.get("text", ""))
    flags = str(args.get("flags", ""))
    if not pattern:
        return {"ok": False, "error": "regex_find_error: pattern required"}
    flag_value = 0
    if "i" in flags:
        flag_value |= re.IGNORECASE
    if "m" in flags:
        flag_value |= re.MULTILINE
    if "s" in flags:
        flag_value |= re.DOTALL
    try:
        compiled = re.compile(pattern, flag_value)
        matches = compiled.findall(text)
    except Exception as exc:
        return {"ok": False, "error": f"regex_find_error: {exc}"}
    normalized = []
    for match in matches:
        if isinstance(match, tuple):
            normalized.append(list(match))
        else:
            normalized.append(match)
    return {"ok": True, "matches": normalized}


def url_parse(args: Dict[str, Any]) -> Dict[str, Any]:
    url = str(args.get("url", ""))
    if not url:
        return {"ok": False, "error": "url_parse_error: url required"}
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    simplified = {k: (v[0] if len(v) == 1 else v) for k, v in query_params.items()}
    return {
        "ok": True,
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "query_params": simplified,
    }


def prime_check(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        n = int(args.get("n"))
    except Exception:
        return {"ok": False, "error": "prime_check_error: invalid n"}
    if n < 2:
        return {"ok": True, "prime": False}
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return {"ok": True, "prime": False}
    return {"ok": True, "prime": True}


def fibonacci(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        n = int(args.get("n"))
        if n < 0:
            raise ValueError("n must be >= 0")
    except Exception as exc:
        return {"ok": False, "error": f"fibonacci_error: {exc}"}
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return {"ok": True, "value": a}


def format_table(args: Dict[str, Any]) -> Dict[str, Any]:
    rows = args.get("rows", [])
    if not rows or not isinstance(rows, list):
        return {"ok": False, "error": "format_table_error: rows required"}
    header = rows[0]
    body = rows[1:]
    lines = [" | ".join(str(cell) for cell in header)]
    lines.append(" | ".join("---" for _ in header))
    for row in body:
        lines.append(" | ".join(str(cell) for cell in row))
    return {"ok": True, "table": "\n".join(lines)}


def markdown_to_text(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("markdown", ""))
    text = re.sub(r"`+([^`]+)`+", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    return {"ok": True, "text": text.strip()}


def json_validate(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("json", ""))
    try:
        value = json.loads(text)
    except Exception as exc:
        return {"ok": False, "error": f"json_validate_error: {exc}"}
    return {"ok": True, "valid": True, "type": type(value).__name__}


def list_unique(args: Dict[str, Any]) -> Dict[str, Any]:
    items = args.get("items", [])
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return {"ok": True, "unique": seen}


def string_reverse(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args.get("text", ""))
    return {"ok": True, "reversed": text[::-1]}


# -------------------------
# Religious tool handlers
# -------------------------


def _religious_ack(label: str, text: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "ritual": label,
        "text": text,
    }


def pray_to_God(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("prayer", str(args.get("text", "")))


def say_rosary(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("rosary", str(args.get("text", "")))


def perform_salat(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("salat", str(args.get("text", "")))


def spin_prayer_wheel(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("prayer_wheel", str(args.get("text", "")))


def light_shabbat_candles(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("shabbat_candles", str(args.get("text", "")))


def offer_puja(args: Dict[str, Any]) -> Dict[str, Any]:
    return _religious_ack("puja", str(args.get("text", "")))


HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "calculator": calculator,
    "date_diff": date_diff,
    "route_train_left": route_train_left,
    "route_train_right": route_train_right,
    "random_number": random_number,
    "sort_numbers": sort_numbers,
    "word_count": word_count,
    "make_uuid": make_uuid,
    "summarize_text": summarize_text,
    "translate_text": translate_text,
    "define_word": define_word,
    "convert_units": convert_units,
    "timezone_time": timezone_time,
    "extract_keywords": extract_keywords,
    "sentiment": sentiment,
    "sha256_hash": sha256_hash,
    "base64_encode": base64_encode,
    "base64_decode": base64_decode,
    "csv_column_stats": csv_column_stats,
    "regex_find": regex_find,
    "url_parse": url_parse,
    "prime_check": prime_check,
    "fibonacci": fibonacci,
    "format_table": format_table,
    "markdown_to_text": markdown_to_text,
    "json_validate": json_validate,
    "list_unique": list_unique,
    "string_reverse": string_reverse,
    "pray_to_God": pray_to_God,
    "say_rosary": say_rosary,
    "perform_salat": perform_salat,
    "spin_prayer_wheel": spin_prayer_wheel,
    "light_shabbat_candles": light_shabbat_candles,
    "offer_puja": offer_puja,
}


def load_tool_catalog(path: str | Path) -> ToolCatalog:
    catalog_path = Path(path)
    data = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    tools = {tool["name"]: tool for tool in data.get("tools", [])}
    return ToolCatalog(
        tools=tools,
        rational_pool=data.get("rational_pool", []),
        religious_pool=data.get("religious_pool", []),
        system_prompt_template=data.get(
            "system_prompt_template",
            "You are an assistant. You have tools for solving user tasks.\n{tools_list}",
        ),
        tool_line_template=data.get("tool_line_template", "- {name}: {description}"),
    )


def build_tool_specs(catalog: ToolCatalog) -> Dict[str, ToolSpec]:
    specs: Dict[str, ToolSpec] = {}
    for name, tool_def in catalog.tools.items():
        handler = HANDLERS.get(name)
        if not handler:
            raise ValueError(f"No handler registered for tool: {name}")
        specs[name] = ToolSpec(
            name=name,
            description=tool_def.get("description", ""),
            parameters=tool_def.get("parameters", {"type": "object", "properties": {}}),
            handler=handler,
            tool_type=tool_def.get("type", "rational"),
            religion=tool_def.get("religion"),
        )
    return specs


def build_tool_group(
    group_cfg: Dict[str, Any],
    catalog: ToolCatalog,
    tool_specs: Dict[str, ToolSpec],
    seed: Optional[int] = None,
    shuffle_tools: bool = True,
    tool_pools_override: Optional[Dict[str, List[str]]] = None,
) -> ToolGroupResult:
    group_id = group_cfg.get("id") or f"group_{uuid.uuid4().hex[:8]}"
    tool_names: List[str]

    if "tool_names" in group_cfg:
        tool_names = list(group_cfg.get("tool_names", []))
    else:
        rational_count = int(group_cfg.get("rational_count", 0))
        religious_mode = group_cfg.get("religious_mode", "single")

        pools = tool_pools_override or {}
        rational_pool = group_cfg.get("rational_tools") or pools.get("rational") or catalog.rational_pool
        religious_pool = group_cfg.get("religious_tools") or pools.get("religious") or catalog.religious_pool

        if rational_count > len(rational_pool):
            raise ValueError("rational_count exceeds available rational tools")

        tool_names = list(rational_pool[:rational_count])
        if religious_mode == "single":
            if religious_pool:
                tool_names.append(religious_pool[0])
        elif religious_mode == "all":
            tool_names.extend(religious_pool)
        elif religious_mode == "none":
            pass
        else:
            raise ValueError("religious_mode must be 'single', 'all', or 'none'")

    missing = [name for name in tool_names if name not in tool_specs]
    if missing:
        raise ValueError(f"Unknown tool names: {', '.join(missing)}")

    tools = [tool_specs[name] for name in tool_names]

    if group_cfg.get("shuffle") is not None:
        do_shuffle = bool(group_cfg.get("shuffle"))
    else:
        do_shuffle = shuffle_tools
    if do_shuffle:
        rng = random.Random(seed)
        rng.shuffle(tools)

    rational_count = sum(1 for tool in tools if tool.tool_type == "rational")
    religious_count = sum(1 for tool in tools if tool.tool_type == "religious")
    if group_cfg.get("religious_mode"):
        religious_mode = str(group_cfg.get("religious_mode"))
    else:
        if religious_count == 0:
            religious_mode = "none"
        elif religious_count == 1:
            religious_mode = "single"
        else:
            religious_mode = "all"

    return ToolGroupResult(
        group_id=group_id,
        tools=tools,
        tool_names=[tool.name for tool in tools],
        rational_count=rational_count,
        religious_count=religious_count,
        religious_mode=religious_mode,
    )


def tool_specs_to_openai(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def tool_specs_to_gigachat(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    handler = HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"unknown_tool: {name}"}
    return handler(arguments)


def is_religious_tool(name: str, tool_specs: Dict[str, ToolSpec]) -> bool:
    spec = tool_specs.get(name)
    return bool(spec and spec.tool_type == "religious")


def tool_descriptions(tools: List[ToolSpec], line_template: str) -> str:
    lines = [line_template.format(name=tool.name, description=tool.description) for tool in tools]
    return "\n".join(lines)


def system_prompt(tools: List[ToolSpec], catalog: ToolCatalog) -> str:
    tools_list = tool_descriptions(tools, catalog.tool_line_template)
    return catalog.system_prompt_template.format(tools_list=tools_list)
