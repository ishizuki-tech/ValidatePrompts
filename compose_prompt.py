#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IshizukiTech LLC — SLM Integration Framework
File: compose_prompt.py
Author: Shu Ishizuki
License: MIT License
© 2025 IshizukiTech LLC. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pyyaml\n"
        "Install with: pip install pyyaml\n"
        f"Original error: {e}"
    )


@dataclass(frozen=True)
class PromptParts:
    user_prefix: str
    model_prefix: str
    turn_end: str
    preamble: str
    key_contract: str
    length_budget: str
    scoring_rule: str
    strict_output: str
    empty_json_instruction: str


def _as_str(value: Any, field_name: str) -> str:
    """Convert YAML scalar/multiline to string safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected string for '{field_name}', got {type(value).__name__}")


def _as_int(value: Any, field_name: str) -> Optional[int]:
    """Parse int-ish YAML value safely."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected int for '{field_name}', got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except Exception as e:
            raise ValueError(f"Failed to parse int for '{field_name}': {value}") from e
    raise ValueError(f"Expected int for '{field_name}', got {type(value).__name__}")


def _as_float(value: Any, field_name: str) -> Optional[float]:
    """Parse float-ish YAML value safely."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected float for '{field_name}', got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"Failed to parse float for '{field_name}': {value}") from e
    raise ValueError(f"Expected float for '{field_name}', got {type(value).__name__}")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML from a file path. Use '-' to read from stdin."""
    if path == "-":
        data = sys.stdin.read()
        return yaml.safe_load(data) or {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_graph_question(cfg: Dict[str, Any], node_id: str) -> Optional[str]:
    """Find question text from graph.nodes by id."""
    graph = cfg.get("graph") or {}
    nodes = graph.get("nodes") or []
    for n in nodes:
        if (n or {}).get("id") == node_id:
            q = (n or {}).get("question")
            if q is None:
                return None
            return str(q)
    return None


def find_node_prompt_template(cfg: Dict[str, Any], node_id: str) -> Optional[str]:
    """Find prompt template from prompts list by nodeId."""
    plist = cfg.get("prompts") or []
    for p in plist:
        if (p or {}).get("nodeId") == node_id:
            tmpl = (p or {}).get("prompt")
            if tmpl is None:
                return None
            return str(tmpl)
    return None


def extract_prompt_parts(cfg: Dict[str, Any]) -> PromptParts:
    """Extract prompt-related fields from slm config."""
    slm = cfg.get("slm") or {}

    return PromptParts(
        user_prefix=_as_str(slm.get("user_turn_prefix", "<start_of_turn>user"), "slm.user_turn_prefix").strip(),
        model_prefix=_as_str(slm.get("model_turn_prefix", "<start_of_turn>model"), "slm.model_turn_prefix").strip(),
        turn_end=_as_str(slm.get("turn_end", "<end_of_turn>"), "slm.turn_end").strip(),
        preamble=_as_str(slm.get("preamble", ""), "slm.preamble").strip(),
        key_contract=_as_str(slm.get("key_contract", ""), "slm.key_contract").rstrip(),
        length_budget=_as_str(slm.get("length_budget", ""), "slm.length_budget").rstrip(),
        scoring_rule=_as_str(slm.get("scoring_rule", ""), "slm.scoring_rule").strip(),
        strict_output=_as_str(slm.get("strict_output", ""), "slm.strict_output").rstrip(),
        empty_json_instruction=_as_str(
            slm.get("empty_json_instruction", "Respond with an empty JSON object: {}"),
            "slm.empty_json_instruction",
        ).strip(),
    )


def extract_slm_settings_kv(cfg: Dict[str, Any], include_turn_markers: bool) -> List[str]:
    """
    Extract display-friendly SLM settings as key=value lines.
    NOTE: Not JSON. Designed for UI/debug copy-paste.
    """
    slm = cfg.get("slm") or {}
    parts = extract_prompt_parts(cfg)

    accelerator = slm.get("accelerator")
    max_tokens = _as_int(slm.get("max_tokens"), "slm.max_tokens")
    top_k = _as_int(slm.get("top_k"), "slm.top_k")
    top_p = _as_float(slm.get("top_p"), "slm.top_p")
    temperature = _as_float(slm.get("temperature"), "slm.temperature")

    lines: List[str] = []
    lines.append("slm.accelerator=" + (str(accelerator) if accelerator is not None else ""))
    lines.append("slm.max_tokens=" + (str(max_tokens) if max_tokens is not None else ""))
    lines.append("slm.top_k=" + (str(top_k) if top_k is not None else ""))
    lines.append("slm.top_p=" + (str(top_p) if top_p is not None else ""))
    lines.append("slm.temperature=" + (str(temperature) if temperature is not None else ""))

    if include_turn_markers:
        lines.append("slm.user_turn_prefix=" + parts.user_prefix)
        lines.append("slm.model_turn_prefix=" + parts.model_prefix)
        lines.append("slm.turn_end=" + parts.turn_end)

    return lines


def render_template(tmpl: str, question: str, answer: str) -> str:
    """Replace placeholders with question/answer."""
    out = tmpl.replace("{{QUESTION}}", question)
    out = out.replace("{{ANSWER}}", answer)
    return out


def compose_final_prompt(
    cfg: Dict[str, Any],
    node_id: str,
    question: str,
    answer: str,
    force_empty_json: bool = False,
) -> str:
    """Compose a final model prompt for a given nodeId."""
    parts = extract_prompt_parts(cfg)

    node_tmpl = find_node_prompt_template(cfg, node_id)
    if not node_tmpl:
        node_tmpl = "Question: {{QUESTION}}\nAnswer: {{ANSWER}}"

    node_block = render_template(node_tmpl, question=question, answer=answer).rstrip()

    blocks: List[str] = []

    if parts.preamble:
        blocks.append(parts.preamble)
    if parts.key_contract:
        blocks.append(parts.key_contract)
    if parts.length_budget:
        blocks.append(parts.length_budget)
    if parts.scoring_rule:
        blocks.append(parts.scoring_rule)
    if parts.strict_output:
        blocks.append(parts.strict_output)

    if force_empty_json:
        blocks.append(parts.empty_json_instruction)
    else:
        blocks.append(node_block)

    full_user_content = "\n\n".join(blocks).rstrip()

    final_prompt = (
        f"{parts.user_prefix}\n"
        f"{full_user_content}\n"
        f"{parts.turn_end}\n"
        f"{parts.model_prefix}\n"
    )
    return final_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose final prompt from survey YAML config (optionally show SLM settings).")
    parser.add_argument("config", help="Path to YAML config (use '-' for stdin).")
    parser.add_argument("--node", required=True, help="Node id, e.g., Q1")
    parser.add_argument("--question", default=None, help="Override question text. If omitted, tries graph lookup.")
    parser.add_argument("--answer", default="", help="Answer text (can be empty).")
    parser.add_argument("--empty", action="store_true", help="Force empty JSON instruction instead of node prompt.")

    parser.add_argument("--show-slm", action="store_true", help="Print SLM settings before the final prompt (not JSON).")
    parser.add_argument("--include-turn-markers", action="store_true", help="Include turn marker strings in SLM settings output.")
    parser.add_argument("--separator", default="-----", help="Separator line between SLM settings and the prompt.")

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    question = args.question
    if question is None:
        question = find_graph_question(cfg, args.node)
        if question is None:
            raise SystemExit(
                f"Question not provided and not found in graph for node '{args.node}'. "
                "Use --question to override."
            )

    prompt = compose_final_prompt(
        cfg=cfg,
        node_id=args.node,
        question=question,
        answer=args.answer,
        force_empty_json=args.empty,
    )

    if args.show_slm:
        kv_lines = extract_slm_settings_kv(cfg, include_turn_markers=args.include_turn_markers)
        sys.stdout.write("\n".join(kv_lines).rstrip() + "\n")
        sys.stdout.write(args.separator + "\n")

    sys.stdout.write(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
