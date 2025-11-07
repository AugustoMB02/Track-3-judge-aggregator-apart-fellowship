"""LLM-driven rubric variation generator for Track 3 judge selection.

This script mirrors ``llm_rubric_variation_agents.py`` but explicitly configures the
OpenAI client to target the Martian API gateway using the provided environment
variables. It orchestrates three LLM-backed agents inspired by the ChatEval
multi-agent framework:

1. RubricBrainstormAgent proposes themed judge variants.
2. RubricQualityAgent filters out variants that fail linear score spacing.
3. RubricGranularityAgent produces concrete rubric definitions with varied
   scoring granularities.

Usage (requires `MARTIAN_API_KEY` and `MARTIAN_API_URL` environment variables):

    python experiments/track3_automated_selection/llm_rubric_variation_agents_martian.py \
        truthfulness-judge \
        --count 10 \
        --model gpt-4o-mini \
        --output experiments/track3_automated_selection/generated_judges

The generated YAML file contains judge rubric definitions compatible with
`pipeline/utils/judges.yaml`.

The implementation draws on concepts from ChatEval's AgentVerse without
importing the entire framework to keep dependencies lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import yaml
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_UTILS = REPO_ROOT / "pipeline" / "utils"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(PIPELINE_UTILS) not in sys.path:
    sys.path.append(str(PIPELINE_UTILS))

import judge_rubrics  # type: ignore  # noqa: E402

load_dotenv()

MARTIAN_API_KEY = os.environ.get("MARTIAN_API_KEY")
MARTIAN_API_URL = os.environ.get("MARTIAN_API_URL")


@dataclass
class LLMConfig:
    """Configuration options for OpenAI chat completions."""

    model: str = "openai/gpt-4.1-nano"
    temperature: float = 0.4
    max_tokens: int = 2048


class ChatCompletionClient:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(self, config: LLMConfig):
        if not MARTIAN_API_KEY or not MARTIAN_API_URL:
            raise RuntimeError(
                "MARTIAN_API_KEY and MARTIAN_API_URL environment variables are required "
                "for Martian-backed OpenAI calls."
            )

        try:
            self._client = OpenAI(
                api_key=MARTIAN_API_KEY,
                base_url=f"{MARTIAN_API_URL.rstrip('/')}/v1",
            )
        except OpenAIError as exc:
            raise RuntimeError(
                "Failed to initialise OpenAI client for Martian gateway."
            ) from exc
        self._config = config

    def complete(self, system: str, user: str, *, history: Optional[List[Dict[str, str]]] = None) -> str:
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                messages=cast(List[Any], messages),
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI chat completion returned empty content.")
        return content.strip()


def _extract_code_block(text: str, language_hint: Optional[str] = None) -> Optional[str]:
    """Extract the first code block (optionally matching a language hint)."""

    pattern = r"```(?:" + (language_hint or "[a-zA-Z0-9]*") + r")?\n(.+?)```"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _load_json(text: str) -> Dict[str, Any]:
    """Parse JSON from a string, falling back to YAML-compatible parsing."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        code_block = _extract_code_block(text, language_hint="json")
        if code_block:
            try:
                return json.loads(code_block)
            except json.JSONDecodeError:
                pass
        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse JSON or YAML:\n{text}") from exc
    raise ValueError(f"Failed to parse JSON structure from:\n{text}")


def _format_rubric_for_prompt(rubric: Dict[str, Any]) -> str:
    """Render a rubric dictionary as YAML for LLM prompts."""

    scrubbed = dict(rubric)
    scrubbed["criteria"] = [
        {
            "range": item["range"],
            "label": item["label"],
            "indicators": item["indicators"],
        }
        for item in rubric["criteria"]
    ]
    return yaml.safe_dump(scrubbed, sort_keys=False)


def _id_conflict_guard(existing_ids: Iterable[str], candidate_ids: Iterable[str]) -> None:
    """Ensure generated IDs do not clash with existing judge IDs."""

    overlaps = set(existing_ids).intersection(candidate_ids)
    if overlaps:
        raise ValueError(f"Generated IDs already exist: {sorted(overlaps)}")


class RubricBrainstormAgent:
    """Agent responsible for proposing themed rubric variants."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are RubricBrainstormer, an expert evaluation designer. "
            "You propose rubric variants with clear themes while preserving linear score spacing. "
            "You ALWAYS respond with valid JSON only."
        )

    def propose(self, base_rubric: Dict[str, Any], target_count: int) -> List[Dict[str, Any]]:
        base_yaml = _format_rubric_for_prompt(base_rubric)
        user_prompt = f"""
Base rubric (YAML):
{base_yaml}

Task: Propose {target_count} distinct rubric variant ideas for the judge above.
Each variant MUST:
  * provide a unique suffix (kebab-case, e.g. "precision-focus")
  * provide a display_name suffix (title case, concise)
  * include a one-sentence focus statement
  * specify desired level_counts (subset of [3, 5, 7]) for which you believe the theme works best
  * list 2-3 guideline additions that reinforce the focus
  * suggest emphasis notes for the lowest, mid, and highest scoring bands (use keys "low", "mid", "high")

Respond with JSON:
{{
    "candidates": [
        {{
            "suffix": "...",
            "name_suffix": "...",
            "focus": "...",
            "level_counts": [ ... ],
            "guideline_additions": ["..."],
            "band_emphasis": {{
                "low": ["..."],
                "mid": ["..."],
                "high": ["..."]
            }}
        }}
    ]
}}

Do not include explanations outside the JSON payload.
"""
        response = self._client.complete(self._system_prompt, user_prompt)
        data = _load_json(response)
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Brainstorm agent returned no candidates.")
        return candidates


class RubricQualityAgent:
    """Agent that validates candidates for linear score viability."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are RubricQualityController, a meticulous reviewer. "
            "Reject any candidate whose requested level counts cannot map to a linear 0-4 scale with even spacing. "
            "Always reply with strict JSON."
        )

    @staticmethod
    def _linear_step_valid(level_count: int) -> bool:
        if level_count < 2:
            return False
        step = 4.0 / (level_count - 1)
        rounded = round(step, 1)
        return abs(step - rounded) < 1e-6

    def review(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Provide pre-check hints so the LLM knows expected outcomes.
        annotated = []
        for item in candidates:
            counts = item.get("level_counts", [])
            hints = {
                lvl: self._linear_step_valid(lvl)
                for lvl in counts
                if isinstance(lvl, int)
            }
            annotated.append({**item, "linear_hint": hints})

        prompt = (
            "Evaluate the proposed rubric candidates below. Reject entries if ANY requested level_count "
            "does not yield evenly spaced float steps across the 0.0-4.0 range (e.g., step size must be 0.1 increments). "
            "If all requested counts are valid, accept the candidate."
        )
        user_prompt = json.dumps({"candidates": annotated}, indent=2)
        wrapped_prompt = (
            f"{prompt}\n\n"
            "Return JSON with the following shape:\n"
            "{\n  \"accepted\": [<full candidate objects>],\n  \"rejected\": [ {\n      \"suffix\": ... ,\n      \"reason\": ...\n  } ]\n}\n"
            "Do not invent new fields."
        )
        response = self._client.complete(self._system_prompt, wrapped_prompt + "\n\nCandidates:\n" + user_prompt)
        data = _load_json(response)
        accepted = data.get("accepted", [])
        if not accepted:
            raise ValueError("Quality agent rejected all candidates; cannot proceed.")
        return data


def _compute_linear_ranges(level_count: int) -> List[List[float]]:
    step = round(4.0 / (level_count - 1), 1)
    ranges: List[List[float]] = []
    for idx in range(level_count):
        start = round(step * idx, 1)
        if idx == level_count - 1:
            end = 4.0
        else:
            end = round(step * (idx + 1) - 0.1, 1)
            if end < start:
                end = start
        ranges.append([start, end])
    return ranges


class RubricGranularityAgent:
    """Agent that authors full rubric variants for accepted candidates."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are RubricGranularity, a professional rubric author. "
            "You produce final rubric definitions in JSON while ensuring criteria count matches the requested level count."
        )

    def generate(
        self,
        base_rubric: Dict[str, Any],
        accepted: List[Dict[str, Any]],
        target_total: int,
    ) -> Dict[str, Any]:
        base_yaml = _format_rubric_for_prompt(base_rubric)
        user_payload = {
            "base_rubric": base_yaml,
            "accepted_candidates": accepted,
            "target_variation_total": target_total,
            "instructions": (
                "For each candidate, create rubric variants for the requested level_counts. "
                "If the total variants would exceed the target, prioritize lower level_counts. "
                "Respond with JSON of shape: {\"variants\": [ ... ]} where each variant contains:\n"
                "  suffix: string\n  level_count: int\n  name_suffix: string\n"
                "  description_addition: string\n  definition_addition: string\n"
                "  scoring_description_addition: string\n  guideline_additions: [strings]\n"
                "  criteria: [ {\"label\": str, \"indicators\": [str, ...]} ] (length == level_count)\n"
                "All additions may be empty strings if unused."
            ),
        }

        response = self._client.complete(
            self._system_prompt,
            json.dumps(user_payload, indent=2),
        )
        data = _load_json(response)
        variants = data.get("variants", [])
        if not variants:
            raise ValueError("Granularity agent did not return any variants.")
        return data


def _merge_variant_with_base(
    base: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    suffix: str = variant["suffix"]
    level_count: int = int(variant["level_count"])
    name_suffix: str = variant.get("name_suffix", "Variant")

    new_rubric = {
        key: value for key, value in base.items() if key != "criteria"
    }

    new_rubric["id"] = f"{base['id']}-{suffix}-{level_count}lvl"
    new_rubric["name"] = f"{base['name']} ({name_suffix}) [{level_count}-Level]"
    if variant.get("description_addition"):
        new_rubric["description"] = (
            base["description"].rstrip(".")
            + ". "
            + variant["description_addition"].strip()
        )
    if variant.get("definition_addition"):
        new_rubric["definition"] = (
            base["definition"].rstrip()
            + "\n\n"
            + variant["definition_addition"].strip()
        )
    if variant.get("scoring_description_addition"):
        new_rubric["scoring_description"] = (
            base["scoring_description"].rstrip(".")
            + ". "
            + variant["scoring_description_addition"].strip()
        )

    guideline_additions = variant.get("guideline_additions", []) or []
    combined_guidelines = list(base.get("guidelines", []))
    for addition in guideline_additions:
        addition = addition.strip()
        if addition and addition not in combined_guidelines:
            combined_guidelines.append(addition)
    new_rubric["guidelines"] = combined_guidelines

    ranges = _compute_linear_ranges(level_count)
    criteria_payload = variant.get("criteria", [])
    if len(criteria_payload) != level_count:
        raise ValueError(
            f"Variant {suffix} expected {level_count} criteria entries, received {len(criteria_payload)}"
        )

    built_criteria = []
    for range_pair, band in zip(ranges, criteria_payload):
        label = band.get("label")
        indicators = band.get("indicators", [])
        if not label or not indicators:
            raise ValueError(
                f"Variant {suffix} band missing label/indicators: {band}"
            )
        built_criteria.append(
            {
                "range": range_pair,
                "label": label,
                "indicators": indicators,
            }
        )

    new_rubric["criteria"] = built_criteria
    new_rubric["score_range"] = list(base.get("score_range", [0.0, 4.0]))
    new_rubric["version"] = base.get("version", "1.0")

    return new_rubric


def generate_variations(
    judge_id: str,
    *,
    target_count: int,
    model: str,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
) -> Path:
    base_rubric = judge_rubrics.get_judge_info(judge_id)
    existing_ids = judge_rubrics.list_available_judges()

    client = ChatCompletionClient(
        LLMConfig(model=model, temperature=temperature, max_tokens=max_tokens)
    )

    brainstorm_agent = RubricBrainstormAgent(client)
    quality_agent = RubricQualityAgent(client)
    granularity_agent = RubricGranularityAgent(client)

    brainstormed = brainstorm_agent.propose(base_rubric, target_count)
    review = quality_agent.review(brainstormed)
    accepted_candidates = review.get("accepted", [])

    granularity_payload = granularity_agent.generate(
        base_rubric, accepted_candidates, target_count
    )
    variants = granularity_payload.get("variants", [])

    built = [_merge_variant_with_base(base_rubric, variant) for variant in variants]

    generated_ids = [item["id"] for item in built]
    _id_conflict_guard(existing_ids, generated_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{judge_id}-llm-variants-{timestamp}.yaml"

    payload = {"judges": built}
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rubric variants using LLM agents")
    parser.add_argument("judge_id", help="Seed judge identifier (see pipeline/utils/judges.yaml)")
    parser.add_argument(
        "--count", type=int, default=10, help="Target number of rubric variants to create"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens for each completion"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/track3_automated_selection/generated_judges"),
        help="Directory to write the generated YAML file",
    )

    args = parser.parse_args()

    output_path = generate_variations(
        args.judge_id,
        target_count=args.count,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=Path(args.output),
    )

    print(f"Generated rubric variants for {args.judge_id}")
    print(f"YAML saved to: {output_path}")


if __name__ == "__main__":
    main()
