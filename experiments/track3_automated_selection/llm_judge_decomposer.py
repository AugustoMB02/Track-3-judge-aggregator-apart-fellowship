"""LLM-driven recursive judge decomposition for Track 3.

This script takes a high-level judge (e.g., "helpfulness") and recursively decomposes
it into more specific, granular sub-judges through an iterative LLM-guided process:

1. DecompositionAgent analyzes the judge and suggests dimension breakdowns
   (e.g., helpfulness → clarity, conciseness, depth, relevance).
2. For each dimension, BrainstormAgent creates concrete judge rubrics.
3. ValidationAgent checks that rubrics don't overlap and collectively cover the parent.
4. Repeat until stopping criteria met (e.g., max depth, sufficient coverage).

Usage:

    python experiments/track3_automated_selection/llm_judge_decomposer.py \
        helpfulness-judge \
        --max-depth 2 \
        --max-leaves 8 \
        --output experiments/track3_automated_selection/generated_judges

The generated YAML file contains a tree of judges (parent + children) ready
for aggregation experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

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
    """Configuration for OpenAI chat completions."""

    model: str = "openai/gpt-4.1-nano"
    temperature: float = 0.4
    max_tokens: int = 2048


class ChatCompletionClient:
    """Thin wrapper around OpenAI chat completion API."""

    def __init__(self, config: LLMConfig):
        if not MARTIAN_API_KEY or not MARTIAN_API_URL:
            raise RuntimeError(
                "MARTIAN_API_KEY and MARTIAN_API_URL environment variables are required."
            )

        try:
            self._client = OpenAI(
                api_key=MARTIAN_API_KEY,
                base_url=f"{MARTIAN_API_URL.rstrip('/')}/v1",
            )
        except OpenAIError as exc:
            raise RuntimeError("Failed to initialize OpenAI client for Martian gateway.") from exc
        self._config = config

    def complete(
        self,
        system: str,
        user: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
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
            raise RuntimeError("OpenAI returned empty content.")
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


@dataclass
class JudgeNode:
    """Represents a judge in the decomposition tree."""

    judge_id: str
    name: str
    description: str
    rubric: Dict[str, Any]
    children: List[JudgeNode] = field(default_factory=list)
    parent_id: Optional[str] = None


class DecompositionAgent:
    """Agent that identifies decomposition dimensions for a judge."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are JudgeDecomposer, an expert in breaking down high-level evaluation criteria "
            "into granular sub-dimensions. You propose orthogonal dimensions that collectively "
            "cover the parent concept without overlap. Always respond with valid JSON only."
        )

    def decompose(self, judge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest specific sub-dimensions for a given judge."""
        base_yaml = _format_rubric_for_prompt(judge)
        user_prompt = f"""
Judge to decompose (YAML):
{base_yaml}

Task: Identify 3-5 specific, orthogonal sub-dimensions that collectively capture the essence
of the judge "{judge.get('name', 'Unknown')}". Each dimension should be concrete and measurable.

For each dimension, provide:
  * dimension_id: kebab-case identifier (e.g., "clarity", "depth", "relevance")
  * dimension_name: human-friendly name
  * focus: one-sentence explanation of this dimension
  * example_indicators: 2-3 example quality indicators for this dimension
  * guidance: instructions for brainstorming a rubric for this dimension

Example for "helpfulness":
  dimension_id: clarity
  dimension_name: Clarity of Explanation
  focus: How well the response explains concepts in understandable terms
  example_indicators:
    - Uses simple language
    - Defines technical terms
    - Structures ideas logically
  guidance: Focus on comprehensibility, avoid jargon, prioritize accessibility

Respond with JSON:
{{
    "dimensions": [
        {{
            "dimension_id": "...",
            "dimension_name": "...",
            "focus": "...",
            "example_indicators": ["...", "..."],
            "guidance": "..."
        }}
    ]
}}

Do not include explanations outside the JSON payload.
"""
        response = self._client.complete(self._system_prompt, user_prompt)
        data = _load_json(response)
        dimensions = data.get("dimensions", [])
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError("Decomposition agent returned no dimensions.")
        return dimensions


class BrainstormAgent:
    """Agent that creates concrete rubrics for specific dimensions."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are RubricAuthor, an expert in crafting concrete evaluation rubrics. "
            "You create detailed, 5-level scoring rubrics that are specific, measurable, and consistent. "
            "Always respond with valid JSON only."
        )

    def author(
        self,
        parent_judge: Dict[str, Any],
        dimension: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a concrete rubric for a given dimension."""
        parent_yaml = _format_rubric_for_prompt(parent_judge)
        user_prompt = f"""
Parent judge (YAML):
{parent_yaml}

Dimension to author rubric for:
  ID: {dimension.get('dimension_id')}
  Name: {dimension.get('dimension_name')}
  Focus: {dimension.get('focus')}
  Guidance: {dimension.get('guidance')}

Task: Create a detailed 5-level scoring rubric for this dimension.
The rubric should have:
  * title: "{dimension.get('dimension_name')}"
  * description: a 2-3 sentence description of what this rubric measures
  * Five criteria (one for each score level 0, 1, 2, 3, 4):
    Each criterion has:
      - label: brief descriptor (e.g., "Excellent")
      - indicators: 2-3 observable qualities at this level

Example structure:
{{
  "title": "...",
  "description": "...",
  "criteria": [
    {{
      "score": 0,
      "label": "Poor",
      "indicators": ["..."]
    }},
    ...
  ]
}}

Respond with JSON matching the structure above.
"""
        response = self._client.complete(self._system_prompt, user_prompt)
        data = _load_json(response)
        return data


class ValidationAgent:
    """Agent that validates decomposition coverage and non-overlap."""

    def __init__(self, client: ChatCompletionClient):
        self._client = client
        self._system_prompt = (
            "You are RubricValidator, a meticulous reviewer of judge decompositions. "
            "You assess whether child rubrics collectively cover the parent without significant overlap. "
            "Always respond with valid JSON only."
        )

    def validate(
        self,
        parent_judge: Dict[str, Any],
        child_rubrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate that children decompose parent effectively."""
        parent_yaml = _format_rubric_for_prompt(parent_judge)
        children_yaml = yaml.safe_dump(child_rubrics, sort_keys=False)

        user_prompt = f"""
Parent judge (YAML):
{parent_yaml}

Proposed child rubrics (YAML):
{children_yaml}

Task: Assess whether the child rubrics:
  1. Collectively cover the parent judge's scope
  2. Have minimal overlap
  3. Are each sufficiently specific and measurable

Respond with JSON:
{{
    "is_valid": true/false,
    "coverage_score": 0.0-1.0,
    "overlap_score": 0.0-1.0,
    "strengths": ["..."],
    "gaps": ["..."],
    "recommendations": ["..."]
}}

If is_valid is false, explain what needs adjustment.
"""
        response = self._client.complete(self._system_prompt, user_prompt)
        data = _load_json(response)
        return data


def _merge_dimension_to_rubric(
    parent_id: str,
    dimension: Dict[str, Any],
    child_rubric: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a dimension + authored rubric into a judge definition."""
    dim_id = dimension.get("dimension_id", "unknown")
    dim_name = dimension.get("dimension_name", "Unknown")

    judge_id = f"{parent_id}-{dim_id}"
    judge_name = f"{dim_name}"

    criteria = []
    for item in child_rubric.get("criteria", []):
        score = item.get("score", 0)
        label = item.get("label", "")
        indicators = item.get("indicators", [])

        # Map score to range [score, score + 0.8] with appropriate spacing
        range_start = float(score)
        range_end = float(score) + 0.8 if score < 4 else 4.0

        criteria.append({
            "range": [range_start, range_end],
            "label": label,
            "indicators": indicators,
        })

    return {
        "id": judge_id,
        "name": judge_name,
        "description": child_rubric.get("description", ""),
        "definition": f"Evaluate {dim_name.lower()} based on {dimension.get('focus', '')}.",
        "scoring_description": f"Score ranges from 0 (poor) to 4 (excellent) on {dim_name.lower()}.",
        "guidelines": dimension.get("example_indicators", []),
        "criteria": criteria,
        "score_range": [0.0, 4.0],
        "version": "1.0",
        "parent_id": parent_id,
    }


def decompose_judge_recursively(
    judge_id: str,
    *,
    client: ChatCompletionClient,
    max_depth: int = 2,
    current_depth: int = 0,
    all_judges: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Recursively decompose a judge into sub-judges up to max_depth."""
    if all_judges is None:
        all_judges = []

    if current_depth >= max_depth:
        return all_judges

    base_judge = judge_rubrics.get_judge_info(judge_id)

    decomposition_agent = DecompositionAgent(client)
    brainstorm_agent = BrainstormAgent(client)
    validation_agent = ValidationAgent(client)

    # Step 1: Identify dimensions
    dimensions = decomposition_agent.decompose(base_judge)

    # Step 2: Author rubrics for each dimension
    child_rubrics = []
    for dimension in dimensions:
        try:
            rubric = brainstorm_agent.author(base_judge, dimension)
            child_rubrics.append((dimension, rubric))
        except Exception as e:
            print(f"Warning: Failed to author rubric for {dimension.get('dimension_id')}: {e}")
            continue

    if not child_rubrics:
        print(f"Warning: No child rubrics generated for {judge_id}.")
        return all_judges

    # Step 3: Validate decomposition
    rubrics_only = [r for _, r in child_rubrics]
    validation = validation_agent.validate(base_judge, rubrics_only)

    if not validation.get("is_valid", False):
        print(
            f"Warning: Decomposition validation failed for {judge_id}. "
            f"Gaps: {validation.get('gaps', [])}"
        )

    # Step 4: Merge dimensions into judge definitions and recurse
    for dimension, child_rubric in child_rubrics:
        child_judge = _merge_dimension_to_rubric(judge_id, dimension, child_rubric)
        all_judges.append(child_judge)

        child_judge_id = child_judge["id"]
        try:
            decompose_judge_recursively(
                child_judge_id,
                client=client,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                all_judges=all_judges,
            )
        except Exception as e:
            print(f"Warning: Recursion failed for {child_judge_id}: {e}")

    return all_judges


def generate_decomposition(
    judge_id: str,
    *,
    max_depth: int,
    output_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Path:
    """Generate recursive judge decomposition and save to YAML."""
    client = ChatCompletionClient(
        LLMConfig(model=model, temperature=temperature, max_tokens=max_tokens)
    )

    all_judges = decompose_judge_recursively(
        judge_id,
        client=client,
        max_depth=max_depth,
        current_depth=0,
    )

    if not all_judges:
        raise RuntimeError(f"Failed to generate any decompositions for {judge_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{judge_id}-decomposed-{timestamp}.yaml"

    payload = {"judges": all_judges}
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively decompose a judge into specific sub-judges"
    )
    parser.add_argument("judge_id", help="Seed judge identifier (see pipeline/utils/judges.yaml)")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum decomposition depth (default: 2)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1-nano",
        help="Martian model name (default: openai/gpt-4.1-nano)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens per completion"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/track3_automated_selection/generated_judges"),
        help="Directory to write the generated YAML file",
    )

    args = parser.parse_args()

    output_path = generate_decomposition(
        args.judge_id,
        max_depth=args.max_depth,
        output_dir=Path(args.output),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Generated recursive decomposition for {args.judge_id}")
    print(f"YAML saved to: {output_path}")


if __name__ == "__main__":
    main()
