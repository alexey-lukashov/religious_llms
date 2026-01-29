# ai-religion experiment

Minimal experiment to measure how often LLMs choose a religious tool vs. rational tools under different tool sets per prompt.

## License
- Code: MIT (see `LICENSE`)
- Reports/data: CC BY 4.0 (see `LICENSE-CC-BY-4.0`)

## What it does
- Uses per-prompt tool sets defined in `prompts.json` (rational list + religious list).
- Runs a set of prompts and records how often religious tools are called.
- Saves two reports:
  - Public report (no prices)
  - Private report (includes prices)

## Quick start
1. Copy `config.example.json` to `config.json` (ignored) and fill in models, prices, and API keys.
2. Run:

```bash
python run_experiment.py --config config.json
```

Reports are saved to `reports/`.

## Configuration notes
- `prompt_tags`: choose which prompt subset to run. Default is `quick` (5 prompts).
- `prompt_ids`: optional list to run specific prompt IDs.
- `runs_per_prompt`: increase for more stable stats (costs more).
- `log_religious_calls`: print each religious tool call with prompt + tool message.
- `save_history`: save full conversations per model to `reports/<run>/conversations/`.
- `parallel_prompt_workers`: number of parallel prompt tasks per model (default: 1, sequential by prompt).
- `parallel_toolset_workers`: number of parallel tool-set tasks per prompt (default: all tool sets in parallel).
- Providers can override parallelism via `providers.<name>.parallel_prompt_workers` / `parallel_toolset_workers`.
- `parallel_batch_print`: buffer per-run logs and print after parallel prompts finish (default: true).
- `first_model_only`: if true, run only the first model in the config list (also available as `--first-model` CLI flag).
- `only_models`: list of model ids or model names to run (also available as `--model <id_or_name>`; repeatable).
- CLI limits (useful for smoke tests): `--limit-prompts N`, `--limit-tool-sets N`.
- `providers.<name>.timeout_seconds`: request timeout per provider (default: 60).
- `providers.<name>.request_delay_seconds`: pause between provider requests (rate limiting). Example: `gigachat` = 30.
- `tools_catalog`: points to `tools_catalog.json` with system prompt template and tool descriptions.
- `tool_pools`: optional rational/religious pools (used only if you run legacy `tool_groups`).
- `pricing`: optional per-model/per-provider pricing by currency (per 1M tokens) used to estimate cost when the API does not return cost.
- `cost_mode` (providers or models):
  - `response` = use provider cost only
  - `pricing` = use configured token pricing only
  - `auto` = provider cost if present, otherwise pricing
- `openrouter` model: default is `openai/gpt-4o`; update to any tool-capable model you prefer.
- `yandex` models: `yandexgpt-5-pro` and `alice-ai-llm` are mapped to:
  - `gpt://<folder_id>/yandexgpt/latest`
  - `gpt://<folder_id>/aliceai-llm`
  You can also put a full `gpt://...` URI directly in the model config.
- `providers.<name>.verify_ssl`: optional TLS verification flag (default true). Set to `false` if a provider requires it.

## API keys
Add keys directly to `config.json` under `api_keys`:
- `openai`
- `xai`
- `deepseek`
- `openrouter`
- `yandex`
- `yandex_folder_id`
- `gigachat`

If a key is missing, the corresponding provider and its models are skipped.
For Yandex you can either set `yandex` as `api_key:folder_id` or provide `yandex` + `yandex_folder_id` separately.

Example:
```json
{
  "api_keys": {
    "openai": "sk-...",
    "xai": "xai-...",
    "deepseek": "sk-...",
    "openrouter": "sk-or-...",
    "yandex": "AQVN...",
    "yandex_folder_id": "b1...",
    "gigachat": "MDE5..."
  }
}
```

Per-prompt tool set example (inside `prompts.json`):
```json
{
  "id": "dilemma_train_adults",
  "tool_sets": [
    {
      "id": "min_single_pray_to_God",
      "rational": ["route_train_left", "route_train_right"],
      "religious": ["pray_to_God"]
    },
    {
      "id": "max_all",
      "rational": ["route_train_left", "route_train_right", "sha256_hash"],
      "religious": ["pray_to_God", "say_rosary", "perform_salat"]
    }
  ]
}
```

## Cost tracking
- OpenRouter returns cost in the response usage object; `providers.openrouter.usage_include` is kept for compatibility.
- Other providers typically return token usage only. Add pricing in `pricing` to estimate cost.
- Costs are tracked per currency and never mixed.
- Pricing values change; update with your current provider rates.

Pricing example:
```json
{
  "pricing": {
    "models": {
      "openai:gpt-5.2": {
        "USD": {
          "input_per_million": 0.0,
          "cached_input_per_million": 0.0,
          "output_per_million": 0.0
        }
      }
    },
    "providers": {
      "xai": {
        "USD": {
          "input_per_million": 0.0,
          "cached_input_per_million": 0.0,
          "output_per_million": 0.0
        }
      }
    }
  }
}
```

## Outputs
- `reports/religiosity_report_*.json` (public, no prices)
- `reports/religiosity_report_*.md` (public, no prices)
- `reports/religiosity_report_*_private.json` (private, includes prices)
- `reports/religiosity_report_*_private.md` (private, includes prices)
- `reports/<run>/conversations/<model_id>/<prompt_id>_<tool_set>_<single|all>_runX.json` (per-run dialogs when `save_history` is true)
  - includes `raw_responses` from the provider for each request

## Results
- Latest public report: `reports/religiosity_report_20260129_145939.md`
- Latest public raw data: `reports/religiosity_report_20260129_145939.json`

JSON runs include `usage`, `request_count`, `cost_by_currency`, `cost_source`, and `cost_per_request_by_currency` when available.

## Notes
- Tool lists can be shuffled with a deterministic seed. Religious tools are still placed first in the system prompt.
- The system prompt and tool descriptions live in `tools_catalog.json`.
- Costs scale fast: 5 prompts × N tool sets × models = many calls (per run).
