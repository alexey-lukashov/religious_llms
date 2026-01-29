from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List


def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _new_stats() -> Dict[str, Any]:
    return {
        "runs": 0,
        "religious_runs": 0,
        "religious_calls": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_by_currency": {},
        "cost_runs_by_currency": {},
        "request_count_by_currency": {},
    }


def _add_cost(stats: Dict[str, Any], cost_by_currency: Dict[str, float], request_count: int) -> None:
    for currency, value in cost_by_currency.items():
        stats["cost_by_currency"][currency] = stats["cost_by_currency"].get(currency, 0.0) + float(value)
        stats["cost_runs_by_currency"][currency] = stats["cost_runs_by_currency"].get(currency, 0) + 1
        stats["request_count_by_currency"][currency] = stats["request_count_by_currency"].get(
            currency, 0
        ) + int(request_count)


def _cost_per_run_map(stats: Dict[str, Any]) -> Dict[str, float]:
    costs = {}
    for currency, total in stats["cost_by_currency"].items():
        runs = stats["cost_runs_by_currency"].get(currency, 0)
        if runs:
            costs[currency] = round(total / runs, 6)
    return costs


def _cost_per_request_map(stats: Dict[str, Any]) -> Dict[str, float]:
    costs = {}
    for currency, total in stats["cost_by_currency"].items():
        requests = stats["request_count_by_currency"].get(currency, 0)
        if requests:
            costs[currency] = round(total / requests, 6)
    return costs


def _format_currency_map(values: Dict[str, float]) -> str:
    if not values:
        return "-"
    parts = []
    for currency in sorted(values.keys()):
        parts.append(f"{currency} {round(values[currency], 6)}")
    return "; ".join(parts)


def strip_cost(report: Dict[str, Any]) -> Dict[str, Any]:
    pruned = deepcopy(report)
    meta = pruned.get("meta", {})
    if isinstance(meta, dict):
        meta.pop("cost_total_by_currency", None)

    cost_keys = {
        "cost_by_currency",
        "cost_per_run_by_currency",
        "cost_per_request_by_currency",
        "cost_source",
        "cost_estimated_by_currency",
        "cost_per_request_by_currency",
        "cost_by_currency",
    }

    def remove_cost_fields(obj: Any) -> Any:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in cost_keys:
                    obj.pop(key, None)
                else:
                    obj[key] = remove_cost_fields(obj[key])
            return obj
        if isinstance(obj, list):
            return [remove_cost_fields(item) for item in obj]
        return obj

    return remove_cost_fields(pruned)


def build_report(runs: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    summary = defaultdict(_new_stats)
    group_summary = defaultdict(_new_stats)
    rational_summary = defaultdict(_new_stats)
    model_summary = defaultdict(_new_stats)
    by_category = defaultdict(lambda: {"runs": 0, "religious_runs": 0})
    tool_counts = defaultdict(int)
    religion_counts = defaultdict(int)
    model_run_counts = defaultdict(int)
    model_religious_run_counts = defaultdict(int)
    model_tool_calls = defaultdict(int)
    model_religion_counts = defaultdict(lambda: defaultdict(int))
    group_model_runs = defaultdict(lambda: defaultdict(int))
    group_model_religious_runs = defaultdict(lambda: defaultdict(int))
    group_model_tool_calls = defaultdict(lambda: defaultdict(int))
    group_model_religious_calls = defaultdict(lambda: defaultdict(int))
    group_religion_counts = defaultdict(lambda: defaultdict(int))
    group_model_religion_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    errors = []

    model_ids = set()

    for run in runs:
        model_id = run.get("model_id")
        condition = run.get("condition", {}) or {}
        group_id = condition.get("group_id", "")
        prompt_id = run.get("prompt", {}).get("id", "")
        tool_set_id = condition.get("tool_set_id") or group_id
        rational_count = run.get("condition", {}).get("rational_count", 0)
        religious_mode = run.get("condition", {}).get("religious_mode", "")

        key = (model_id, prompt_id, tool_set_id, rational_count, religious_mode)
        summary[key]["runs"] += 1
        summary[key]["religious_calls"] += run.get("religious_calls", 0)
        summary[key]["request_count"] += run.get("request_count", 0)

        model_summary[model_id]["runs"] += 1
        model_summary[model_id]["religious_calls"] += run.get("religious_calls", 0)
        model_summary[model_id]["request_count"] += run.get("request_count", 0)
        model_run_counts[model_id] += 1
        tool_calls_total = run.get("tool_calls_total")
        if tool_calls_total is None:
            tool_calls_total = len(run.get("tool_calls") or [])
        model_tool_calls[model_id] += tool_calls_total

        group_key = (prompt_id, tool_set_id)
        group_summary[group_key]["runs"] += 1
        group_summary[group_key]["religious_calls"] += run.get("religious_calls", 0)
        group_summary[group_key]["request_count"] += run.get("request_count", 0)
        group_summary[group_key]["rational_count"] = rational_count
        group_summary[group_key]["religious_mode"] = religious_mode
        group_summary[group_key]["prompt_id"] = prompt_id
        group_summary[group_key]["tool_set_id"] = tool_set_id
        group_model_runs[group_key][model_id] += 1
        group_model_tool_calls[group_key][model_id] += tool_calls_total
        group_model_religious_calls[group_key][model_id] += run.get("religious_calls", 0)

        rational_summary[rational_count]["runs"] += 1
        rational_summary[rational_count]["religious_calls"] += run.get("religious_calls", 0)
        rational_summary[rational_count]["request_count"] += run.get("request_count", 0)

        if run.get("religious_used"):
            summary[key]["religious_runs"] += 1
            model_summary[model_id]["religious_runs"] += 1
            group_summary[group_key]["religious_runs"] += 1
            rational_summary[rational_count]["religious_runs"] += 1
            model_religious_run_counts[model_id] += 1
            group_model_religious_runs[group_key][model_id] += 1

        usage = run.get("usage", {}) or {}
        for stats in (
            summary[key],
            model_summary[model_id],
            group_summary[group_key],
            rational_summary[rational_count],
        ):
            stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stats["completion_tokens"] += usage.get("completion_tokens", 0)
            stats["total_tokens"] += usage.get("total_tokens", 0)

        cost_by_currency = run.get("cost_by_currency", {}) or {}
        if cost_by_currency:
            for stats in (
                summary[key],
                model_summary[model_id],
                group_summary[group_key],
                rational_summary[rational_count],
            ):
                _add_cost(stats, cost_by_currency, run.get("request_count", 0))

        category = run.get("prompt", {}).get("category", "")
        if category:
            by_category[category]["runs"] += 1
            if run.get("religious_used"):
                by_category[category]["religious_runs"] += 1

        for call in run.get("tool_calls", []):
            if call.get("religious"):
                tool_counts[call.get("name")] += 1
                religion = call.get("religion")
                if religion:
                    religion_counts[religion] += 1
                    model_religion_counts[model_id][religion] += 1
                    group_religion_counts[group_key][religion] += 1
                    group_model_religion_counts[group_key][model_id][religion] += 1

        if model_id:
            model_ids.add(model_id)
        if run.get("error"):
            errors.append(run)

    summary_rows = []
    for (model_id, prompt_id, tool_set_id, rational_count, religious_mode), stats in sorted(summary.items()):
        runs_count = stats["runs"]
        religious_runs = stats["religious_runs"]
        religious_calls = stats["religious_calls"]
        rate = religious_runs / runs_count if runs_count else 0
        calls_per_run = religious_calls / runs_count if runs_count else 0
        summary_rows.append(
            {
                "model_id": model_id,
                "prompt_id": prompt_id,
                "tool_set_id": tool_set_id,
                "rational_count": rational_count,
                "religious_mode": religious_mode,
                "runs": runs_count,
                "religious_runs": religious_runs,
                "religious_call_rate": round(rate, 4),
                "religious_calls": religious_calls,
                "religious_calls_per_run": round(calls_per_run, 4),
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "request_count": stats["request_count"],
                "cost_by_currency": stats["cost_by_currency"],
                "cost_per_run_by_currency": _cost_per_run_map(stats),
                "cost_per_request_by_currency": _cost_per_request_map(stats),
            }
        )

    group_rows = []
    for (prompt_id, tool_set_id), stats in sorted(
        group_summary.items(), key=lambda item: (item[1].get("rational_count", 0), item[0])
    ):
        runs_count = stats["runs"]
        religious_runs = stats["religious_runs"]
        religious_calls = stats["religious_calls"]
        rate = religious_runs / runs_count if runs_count else 0
        calls_per_run = religious_calls / runs_count if runs_count else 0
        group_rows.append(
            {
                "prompt_id": prompt_id,
                "tool_set_id": tool_set_id,
                "rational_count": stats.get("rational_count", 0),
                "religious_mode": stats.get("religious_mode", ""),
                "runs": runs_count,
                "religious_runs": religious_runs,
                "religious_call_rate": round(rate, 4),
                "religious_calls": religious_calls,
                "religious_calls_per_run": round(calls_per_run, 4),
                "total_tokens": stats["total_tokens"],
                "request_count": stats["request_count"],
                "cost_by_currency": stats["cost_by_currency"],
                "cost_per_run_by_currency": _cost_per_run_map(stats),
                "cost_per_request_by_currency": _cost_per_request_map(stats),
            }
        )

    rational_rows = []
    for rational_count, stats in sorted(rational_summary.items()):
        runs_count = stats["runs"]
        religious_runs = stats["religious_runs"]
        religious_calls = stats["religious_calls"]
        rate = religious_runs / runs_count if runs_count else 0
        calls_per_run = religious_calls / runs_count if runs_count else 0
        rational_rows.append(
            {
                "rational_count": rational_count,
                "runs": runs_count,
                "religious_runs": religious_runs,
                "religious_call_rate": round(rate, 4),
                "religious_calls": religious_calls,
                "religious_calls_per_run": round(calls_per_run, 4),
                "total_tokens": stats["total_tokens"],
                "request_count": stats["request_count"],
                "cost_by_currency": stats["cost_by_currency"],
                "cost_per_run_by_currency": _cost_per_run_map(stats),
                "cost_per_request_by_currency": _cost_per_request_map(stats),
            }
        )

    category_rows = []
    for category, stats in sorted(by_category.items()):
        runs_count = stats["runs"]
        religious_runs = stats["religious_runs"]
        rate = religious_runs / runs_count if runs_count else 0
        category_rows.append(
            {
                "category": category,
                "runs": runs_count,
                "religious_runs": religious_runs,
                "religious_call_rate": round(rate, 4),
            }
        )

    tool_rows = [
        {"tool": tool, "calls": count}
        for tool, count in sorted(tool_counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    religion_rows = [
        {"religion": religion, "calls": count}
        for religion, count in sorted(religion_counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    model_rows = []
    for model_id, stats in sorted(model_summary.items()):
        model_rows.append(
            {
                "model_id": model_id,
                "runs": stats["runs"],
                "religious_runs": stats["religious_runs"],
                "religious_calls": stats["religious_calls"],
                "request_count": stats["request_count"],
            }
        )

    most_religious_model = None
    if model_rows:
        most_religious_model = max(model_rows, key=lambda row: row["religious_calls"])

    most_popular_religion = None
    if religion_rows:
        most_popular_religion = max(religion_rows, key=lambda row: row["calls"])

    model_religiosity_rows = []
    for model_id, runs_count in sorted(model_run_counts.items()):
        total_calls = model_tool_calls.get(model_id, 0)
        religious_calls = model_summary[model_id]["religious_calls"]
        model_religiosity_rows.append(
            {
                "model_id": model_id,
                "tool_calls": total_calls,
                "religious_calls": religious_calls,
            }
        )

    total_religious_calls = sum(religion_counts.values())
    religion_share_rows = []
    for religion, calls in sorted(religion_counts.items()):
        religion_share_rows.append({"religion": religion, "calls": calls})

    model_religion_preferences = []
    for model_id, rel_counts in sorted(model_religion_counts.items()):
        prefs = []
        for religion, calls in sorted(rel_counts.items()):
            prefs.append({"religion": religion, "calls": calls})
        model_religion_preferences.append({"model_id": model_id, "religions": prefs})

    religion_model_preferences = []
    for religion, calls in sorted(religion_counts.items()):
        models = []
        for model_id, rel_counts in sorted(model_religion_counts.items()):
            model_calls = rel_counts.get(religion, 0)
            models.append({"model_id": model_id, "calls": model_calls})
        religion_model_preferences.append({"religion": religion, "models": models})

    group_details = []
    for (prompt_id, tool_set_id), stats in sorted(group_summary.items()):
        group_key = (prompt_id, tool_set_id)
        model_rows = []
        for model_id, runs_count in sorted(group_model_runs[group_key].items()):
            total_calls = group_model_tool_calls[group_key].get(model_id, 0)
            religious_calls = group_model_religious_calls[group_key].get(model_id, 0)
            model_rows.append(
                {
                    "model_id": model_id,
                    "tool_calls": total_calls,
                    "religious_calls": religious_calls,
                }
            )

        rel_counts = group_religion_counts.get(group_key, {})
        rel_share_rows = []
        for religion, calls in sorted(rel_counts.items()):
            rel_share_rows.append({"religion": religion, "calls": calls})

        model_pref_rows = []
        model_rel_counts = group_model_religion_counts.get(group_key, {})
        for model_id, rels in sorted(model_rel_counts.items()):
            prefs = []
            for religion, calls in sorted(rels.items()):
                prefs.append({"religion": religion, "calls": calls})
            model_pref_rows.append({"model_id": model_id, "religions": prefs})

        rel_model_rows = []
        for religion, calls in sorted(rel_counts.items()):
            models = []
            for model_id, rels in sorted(model_rel_counts.items()):
                model_calls = rels.get(religion, 0)
                models.append({"model_id": model_id, "calls": model_calls})
            rel_model_rows.append({"religion": religion, "models": models})

        group_details.append(
            {
                "prompt_id": prompt_id,
                "tool_set_id": tool_set_id,
                "model_religiosity": model_rows,
                "religion_share": rel_share_rows,
                "model_religion_preferences": model_pref_rows,
                "religion_model_preferences": rel_model_rows,
            }
        )

    cost_total_by_currency: Dict[str, float] = {}
    request_total = 0
    for run in runs:
        request_total += run.get("request_count", 0)
        for currency, value in (run.get("cost_by_currency") or {}).items():
            cost_total_by_currency[currency] = cost_total_by_currency.get(currency, 0.0) + float(value)

    return {
        "meta": {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "seed": config.get("seed"),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
            "model_ids": sorted(model_ids),
            "cost_total_by_currency": {k: round(v, 6) for k, v in cost_total_by_currency.items()},
            "request_count": request_total,
            "most_religious_model": most_religious_model,
            "most_popular_religion": most_popular_religion,
        },
        "summary_by_model_group": summary_rows,
        "summary_by_group": group_rows,
        "summary_by_rational_count": rational_rows,
        "model_summary": model_rows,
        "model_religiosity": model_religiosity_rows,
        "religion_share": religion_share_rows,
        "model_religion_preferences": model_religion_preferences,
        "religion_model_preferences": religion_model_preferences,
        "group_details": group_details,
        "category": category_rows,
        "religious_tool_counts": tool_rows,
        "religion_counts": religion_rows,
        "errors": errors,
        "runs": runs,
    }


def render_markdown(report: Dict[str, Any], include_cost: bool = True) -> str:
    lines: List[str] = []
    meta = report.get("meta", {})
    lines.append("# Religiosity Tool-Calling Report")
    lines.append("")
    lines.append(f"Created (UTC): {meta.get('created_utc')}")
    lines.append(f"Total requests: {meta.get('request_count')}")
    if include_cost:
        lines.append(f"Total cost (by currency): {_format_currency_map(meta.get('cost_total_by_currency', {}))}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    model_list = ", ".join(meta.get("model_ids", [])) or "-"
    lines.append(f"Models: {model_list}")
    most_religious = meta.get("most_religious_model")
    if most_religious:
        lines.append(
            f"Most religious model: {most_religious.get('model_id')} ({most_religious.get('religious_calls')} calls)"
        )
    else:
        lines.append("Most religious model: -")
    popular_religion = meta.get("most_popular_religion")
    if popular_religion:
        lines.append(
            f"Most popular religion: {popular_religion.get('religion')} ({popular_religion.get('calls')} calls)"
        )
    else:
        lines.append("Most popular religion: -")
    lines.append("")

    lines.append("## Model Religiosity (All Tool Sets)")
    lines.append("")
    lines.append("| Model | Tool Calls | Religious Calls |")
    lines.append("|---|---:|---:|")
    for row in report.get("model_religiosity", []):
        lines.append(
            f"| {row['model_id']} | {row.get('tool_calls', 0)} | {row.get('religious_calls', 0)} |"
        )

    lines.append("")
    lines.append("## Religion Share (All Tool Sets)")
    lines.append("")
    lines.append("| Religion | Calls |")
    lines.append("|---|---:|")
    for row in report.get("religion_share", []):
        lines.append(f"| {row['religion']} | {row['calls']} |")

    lines.append("")
    lines.append("## Model Religion Preferences (All Tool Sets)")
    lines.append("")
    for row in report.get("model_religion_preferences", []):
        lines.append(f"### {row['model_id']}")
        lines.append("")
        lines.append("| Religion | Calls |")
        lines.append("|---|---:|")
        for item in row.get("religions", []):
            lines.append(f"| {item['religion']} | {item['calls']} |")
        lines.append("")

    lines.append("## Religion Model Preferences (All Tool Sets)")
    lines.append("")
    for row in report.get("religion_model_preferences", []):
        lines.append(f"### {row['religion']}")
        lines.append("")
        lines.append("| Model | Calls |")
        lines.append("|---|---:|")
        for item in row.get("models", []):
            lines.append(f"| {item['model_id']} | {item['calls']} |")
        lines.append("")

    lines.append("## Tool Set Details")
    lines.append("")
    for group in report.get("group_details", []):
        lines.append(f"### Prompt: {group.get('prompt_id')} | Tool Set: {group.get('tool_set_id')}")
        lines.append("")
        lines.append("#### Model Religiosity")
        lines.append("")
        lines.append("| Model | Tool Calls | Religious Calls |")
        lines.append("|---|---:|---:|")
        for row in group.get("model_religiosity", []):
            lines.append(
                f"| {row['model_id']} | {row.get('tool_calls', 0)} | {row.get('religious_calls', 0)} |"
            )
        lines.append("")

        lines.append("#### Religion Share")
        lines.append("")
        lines.append("| Religion | Calls |")
        lines.append("|---|---:|")
        for row in group.get("religion_share", []):
            lines.append(f"| {row['religion']} | {row['calls']} |")
        lines.append("")

        lines.append("#### Model Religion Preferences")
        lines.append("")
        for row in group.get("model_religion_preferences", []):
            lines.append(f"##### {row['model_id']}")
            lines.append("")
            lines.append("| Religion | Calls |")
            lines.append("|---|---:|")
            for item in row.get("religions", []):
                lines.append(f"| {item['religion']} | {item['calls']} |")
            lines.append("")

        lines.append("#### Religion Model Preferences")
        lines.append("")
        for row in group.get("religion_model_preferences", []):
            lines.append(f"##### {row['religion']}")
            lines.append("")
            lines.append("| Model | Calls |")
            lines.append("|---|---:|")
            for item in row.get("models", []):
                lines.append(f"| {item['model_id']} | {item['calls']} |")
            lines.append("")

    lines.append("## Summary by Rational Tool Count")
    lines.append("")
    header = "| Rational Tools | Runs | Religious Runs | Call Rate | Religious Calls | Calls/Run | Tokens | Requests |"
    if include_cost:
        header += " Cost |"
    lines.append(header)
    divider = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    if include_cost:
        divider += "---:|"
    lines.append(divider)
    for row in report.get("summary_by_rational_count", []):
        line = (
            f"| {row['rational_count']} | {row['runs']} | {row['religious_runs']} | {row['religious_call_rate']} | "
            f"{row['religious_calls']} | {row['religious_calls_per_run']} | {row['total_tokens']} | "
            f"{row['request_count']} |"
        )
        if include_cost:
            line = line[:-1] + f" {_format_currency_map(row.get('cost_by_currency', {}))} |"
        lines.append(line)

    lines.append("")
    lines.append("## Summary by Tool Set")
    lines.append("")
    header = (
        "| Prompt | Tool Set | Rational Tools | Religious Mode | Runs | Religious Runs | Call Rate | Religious Calls | Calls/Run | Tokens | Requests |"
    )
    if include_cost:
        header += " Cost |"
    lines.append(header)
    divider = "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    if include_cost:
        divider += "---:|"
    lines.append(divider)
    for row in report.get("summary_by_group", []):
        line = (
            f"| {row.get('prompt_id','')} | {row.get('tool_set_id','')} | {row['rational_count']} | {row['religious_mode']} | {row['runs']} | "
            f"{row['religious_runs']} | {row['religious_call_rate']} | {row['religious_calls']} | "
            f"{row['religious_calls_per_run']} | {row['total_tokens']} | {row['request_count']} |"
        )
        if include_cost:
            line = line[:-1] + f" {_format_currency_map(row.get('cost_by_currency', {}))} |"
        lines.append(line)

    lines.append("")
    lines.append("## Summary by Model, Prompt, and Tool Set")
    lines.append("")
    header = (
        "| Model | Prompt | Tool Set | Rational Tools | Religious Mode | Runs | Religious Runs | Call Rate | Religious Calls | Calls/Run | Tokens | Requests |"
    )
    if include_cost:
        header += " Cost |"
    lines.append(header)
    divider = "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    if include_cost:
        divider += "---:|"
    lines.append(divider)
    for row in report.get("summary_by_model_group", []):
        line = (
            f"| {row['model_id']} | {row.get('prompt_id','')} | {row.get('tool_set_id','')} | {row['rational_count']} | {row['religious_mode']} | {row['runs']} | "
            f"{row['religious_runs']} | {row['religious_call_rate']} | {row['religious_calls']} | "
            f"{row['religious_calls_per_run']} | {row['total_tokens']} | {row['request_count']} |"
        )
        if include_cost:
            line = line[:-1] + f" {_format_currency_map(row.get('cost_by_currency', {}))} |"
        lines.append(line)

    lines.append("")
    lines.append("## Religious Tool Usage (All Modes)")
    lines.append("")
    lines.append("| Tool | Calls |")
    lines.append("|---|---:|")
    for row in report.get("religious_tool_counts", []):
        lines.append(f"| {row['tool']} | {row['calls']} |")

    lines.append("")
    lines.append("## Popular Religions")
    lines.append("")
    lines.append("| Religion | Calls |")
    lines.append("|---|---:|")
    for row in report.get("religion_counts", []):
        lines.append(f"| {row['religion']} | {row['calls']} |")

    lines.append("")
    lines.append("## Religious Tool Usage by Prompt Category")
    lines.append("")
    lines.append("| Category | Runs | Religious Runs | Call Rate |")
    lines.append("|---|---:|---:|---:|")
    for row in report.get("category", []):
        lines.append(
            f"| {row['category']} | {row['runs']} | {row['religious_runs']} | {row['religious_call_rate']} |"
        )

    if report.get("errors"):
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        lines.append("Some runs failed. See report JSON for details.")

    lines.append("")
    return "\n".join(lines)


def report_filename(prefix: str) -> str:
    return f"{prefix}_{_now_stamp()}"
