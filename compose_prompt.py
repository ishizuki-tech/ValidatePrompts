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
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pyyaml\n"
        "Install with: pip install pyyaml\n"
        f"Original error: {e}"
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class PromptParts:
    """Small container for SLM prompt contract pieces loaded from config."""
    user_prefix: str
    model_prefix: str
    turn_end: str
    preamble: str
    key_contract: str
    length_budget: str
    scoring_rule: str
    strict_output: str
    empty_json_instruction: str


# =============================================================================
# Helpers (errors, coercion, IO)
# =============================================================================

def _die(msg: str, code: int = 2) -> None:
    """Exit with a human-friendly error message and an exit code."""
    sys.stderr.write(f"[compose_prompt] ERROR: {msg}\n")
    raise SystemExit(code)


def _as_str(value: Any, field_name: str, default: Optional[str] = None) -> str:
    """Coerce a config value to string, or use a default."""
    if value is None:
        if default is None:
            _die(f"Missing required field: {field_name}")
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _as_int(value: Any, field_name: str, default: Optional[int] = None) -> int:
    """Coerce a config value to int, or use a default."""
    if value is None:
        if default is None:
            _die(f"Missing required int field: {field_name}")
        return default
    if isinstance(value, bool):
        _die(f"Invalid int field (bool not allowed): {field_name}")
    try:
        return int(value)
    except Exception:
        _die(f"Invalid int value for {field_name}: {value}")


def _as_float(value: Any, field_name: str, default: Optional[float] = None) -> float:
    """Coerce a config value to float, or use a default."""
    if value is None:
        if default is None:
            _die(f"Missing required float field: {field_name}")
        return default
    if isinstance(value, bool):
        _die(f"Invalid float field (bool not allowed): {field_name}")
    try:
        return float(value)
    except Exception:
        _die(f"Invalid float value for {field_name}: {value}")


def _read_text_file(path: Path) -> str:
    """Read UTF-8 text content from a file."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _die(f"File not found: {path}")
    except Exception as e:
        _die(f"Failed to read file: {path} ({e})")


def _load_yaml_from_text(raw: str, origin: str) -> Dict[str, Any]:
    """Parse YAML from raw text into a dict."""
    try:
        data = yaml.safe_load(raw)
    except Exception as e:
        _die(f"Invalid YAML: {origin} ({e})")
    if not isinstance(data, dict):
        _die(f"Top-level YAML must be a mapping/object: {origin}")
    return data


def _load_yaml_config(path_or_dash: str) -> Dict[str, Any]:
    """
    Load YAML config into a dict.

    Rules:
    - If path is '-', read YAML from stdin.
    - Otherwise, read from the given file path.
    """
    if path_or_dash.strip() == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            _die("No YAML content provided on stdin ('-')")
        return _load_yaml_from_text(raw, origin="stdin")
    p = Path(path_or_dash)
    raw = _read_text_file(p)
    return _load_yaml_from_text(raw, origin=str(p))


# =============================================================================
# Graph helpers
# =============================================================================

def _get_nodes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return graph nodes list."""
    graph = cfg.get("graph")
    if not isinstance(graph, dict):
        _die("Missing or invalid 'graph' section")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _die("Missing or invalid 'graph.nodes' (must be a list)")
    out: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            _die(f"Invalid node at graph.nodes[{i}] (must be an object)")
        out.append(n)
    return out


def _find_node(cfg: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Find a node by id."""
    for n in _get_nodes(cfg):
        if _as_str(n.get("id"), "graph.nodes[].id", default="") == node_id:
            return n
    _die(f"Node id not found in graph.nodes: {node_id}")


def _list_node_summaries(cfg: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return (id, type, title) for all nodes."""
    out: List[Tuple[str, str, str]] = []
    for n in _get_nodes(cfg):
        nid = _as_str(n.get("id"), "graph.nodes[].id", default="")
        ntype = _as_str(n.get("type"), f"graph.nodes[{nid}].type", default="")
        title = _as_str(n.get("title"), f"graph.nodes[{nid}].title", default="")
        out.append((nid, ntype, title))
    return out


# =============================================================================
# SLM prompt contract + settings dump
# =============================================================================

def _load_parts(cfg: Dict[str, Any]) -> PromptParts:
    """Load SLM prompt parts from cfg['slm']."""
    slm = cfg.get("slm")
    if not isinstance(slm, dict):
        _die("Missing or invalid 'slm' section")

    user_prefix = _as_str(slm.get("user_turn_prefix"), "slm.user_turn_prefix", default="<start_of_turn>user")
    model_prefix = _as_str(slm.get("model_turn_prefix"), "slm.model_turn_prefix", default="<start_of_turn>model")
    turn_end = _as_str(slm.get("turn_end"), "slm.turn_end", default="<end_of_turn>")
    preamble = _as_str(slm.get("preamble"), "slm.preamble", default="")
    key_contract = _as_str(slm.get("key_contract"), "slm.key_contract", default="")
    length_budget = _as_str(slm.get("length_budget"), "slm.length_budget", default="")
    scoring_rule = _as_str(slm.get("scoring_rule"), "slm.scoring_rule", default="")
    strict_output = _as_str(slm.get("strict_output"), "slm.strict_output", default="")
    empty_json_instruction = _as_str(
        slm.get("empty_json_instruction"),
        "slm.empty_json_instruction",
        default="Respond with an empty JSON object: {}",
    )

    return PromptParts(
        user_prefix=user_prefix,
        model_prefix=model_prefix,
        turn_end=turn_end,
        preamble=preamble.strip("\n"),
        key_contract=key_contract.strip("\n"),
        length_budget=length_budget.strip("\n"),
        scoring_rule=scoring_rule.strip("\n"),
        strict_output=strict_output.strip("\n"),
        empty_json_instruction=empty_json_instruction.strip("\n"),
    )


def _collect_slm_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Collect SLM runtime-relevant settings from cfg['slm'] for display/debug."""
    slm = cfg.get("slm")
    if not isinstance(slm, dict):
        _die("Missing or invalid 'slm' section")

    settings: Dict[str, Any] = {
        "accelerator": _as_str(slm.get("accelerator"), "slm.accelerator", default=""),
        "max_tokens": _as_int(slm.get("max_tokens"), "slm.max_tokens", default=0),
        "top_k": _as_int(slm.get("top_k"), "slm.top_k", default=0),
        "top_p": _as_float(slm.get("top_p"), "slm.top_p", default=0.0),
        "temperature": _as_float(slm.get("temperature"), "slm.temperature", default=0.0),
        "user_turn_prefix": _as_str(slm.get("user_turn_prefix"), "slm.user_turn_prefix", default="<start_of_turn>user"),
        "model_turn_prefix": _as_str(slm.get("model_turn_prefix"), "slm.model_turn_prefix", default="<start_of_turn>model"),
        "turn_end": _as_str(slm.get("turn_end"), "slm.turn_end", default="<end_of_turn>"),
        # Prompt contract fields (useful for debugging)
        "preamble": _as_str(slm.get("preamble"), "slm.preamble", default=""),
        "key_contract": _as_str(slm.get("key_contract"), "slm.key_contract", default=""),
        "length_budget": _as_str(slm.get("length_budget"), "slm.length_budget", default=""),
        "scoring_rule": _as_str(slm.get("scoring_rule"), "slm.scoring_rule", default=""),
        "strict_output": _as_str(slm.get("strict_output"), "slm.strict_output", default=""),
        "empty_json_instruction": _as_str(slm.get("empty_json_instruction"), "slm.empty_json_instruction", default="Respond with an empty JSON object: {}"),
    }
    return settings


def _kv_safe(v: Any) -> str:
    """
    Format a value as a single-line string for key=value output.
    - Strings become JSON-escaped (so newlines turn into \\n).
    - Other types become JSON too when possible.
    """
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _format_slm_settings(settings: Dict[str, Any], fmt: str) -> str:
    """Format settings for display (stderr by default)."""
    fmt_norm = (fmt or "kv").lower().strip()
    if fmt_norm == "json":
        return json.dumps(settings, ensure_ascii=False, separators=(",", ":"))
    if fmt_norm == "yaml":
        return yaml.safe_dump({"slm": settings}, sort_keys=False, allow_unicode=True).rstrip("\n")

    lines: List[str] = ["slm settings:"]
    for k, v in settings.items():
        lines.append(f"- {k}={_kv_safe(v)}")
    return "\n".join(lines)


# =============================================================================
# Prompt templates
# =============================================================================

def _find_prompt_template(cfg: Dict[str, Any], node_id: str) -> Optional[str]:
    """Find prompt template text for a given nodeId."""
    prompts = cfg.get("prompts")
    if prompts is None:
        return None
    if not isinstance(prompts, list):
        _die("Invalid 'prompts' section (must be a list)")
    for i, p in enumerate(prompts):
        if not isinstance(p, dict):
            _die(f"Invalid prompts[{i}] (must be an object)")
        pid = _as_str(p.get("nodeId"), f"prompts[{i}].nodeId", default="")
        if pid == node_id:
            tmpl = p.get("prompt")
            if tmpl is None:
                return ""
            return _as_str(tmpl, f"prompts[{i}].prompt", default="")
    return None


def _default_task_prompt(question: str, answer: str) -> str:
    """Fallback prompt when a node-specific template is missing."""
    return (
        "Task:\n"
        "- Note weaknesses (e.g., unclear units, missing baseline/time window).\n"
        "- Provide ONE strong expected answer that fits the question.\n"
        "- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence).\n"
        "- Score 1–100 (integer).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )


def _apply_placeholders(template: str, question: str, answer: str) -> str:
    """Replace {{QUESTION}} and {{ANSWER}} placeholders."""
    out = template.replace("{{QUESTION}}", question).replace("{{ANSWER}}", answer)
    out = out.replace("{{ QUESTION }}", question).replace("{{ ANSWER }}", answer)
    return out


def compose_prompt(cfg: Dict[str, Any], node_id: str, answer: str, question_override: Optional[str] = None) -> str:
    """Compose the final prompt for a given node + answer."""
    parts = _load_parts(cfg)
    node = _find_node(cfg, node_id)

    node_type = _as_str(node.get("type"), f"graph.nodes[{node_id}].type", default="")
    node_question = _as_str(node.get("question"), f"graph.nodes[{node_id}].question", default="")
    question = question_override if (question_override is not None and question_override != "") else node_question

    # For non-AI nodes, return an "empty JSON" instruction prompt (or skip calling the model upstream).
    if node_type.upper() != "AI":
        body = parts.empty_json_instruction
        sections = [
            parts.user_prefix,
            parts.preamble,
            parts.key_contract,
            parts.length_budget,
            parts.scoring_rule,
            parts.strict_output,
            body,
            parts.turn_end,
            parts.model_prefix,
        ]
        return "\n".join([s for s in sections if s.strip() != ""]).strip() + "\n"

    tmpl = _find_prompt_template(cfg, node_id)
    if tmpl is None or tmpl.strip() == "":
        body = _default_task_prompt(question=question, answer=answer)
    else:
        body = _apply_placeholders(tmpl, question=question, answer=answer)

    sections = [
        parts.user_prefix,
        parts.preamble,
        parts.key_contract,
        parts.length_budget,
        parts.scoring_rule,
        parts.strict_output,
        body.strip("\n"),
        parts.turn_end,
        parts.model_prefix,
    ]
    return "\n".join([s for s in sections if s.strip() != ""]).strip() + "\n"


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI args."""
    p = argparse.ArgumentParser(
        prog="compose_prompt.py",
        description="Compose an SLM prompt from survey_config.yaml for a given node and answer.",
    )
    p.add_argument("config", help="Path to survey_config.yaml (use '-' to read from stdin)")
    p.add_argument("--node", required=True, help="Node id (e.g., Q1, Q2, ...)")
    p.add_argument("--answer", required=True, help="Answer text to evaluate")
    p.add_argument("--question", default=None, help="Optional override question text (otherwise uses graph.nodes[].question)")
    p.add_argument("--list-nodes", action="store_true", help="List node ids/types/titles and exit")

    # SLM settings display/debug
    p.add_argument("--slm-format", choices=["kv", "yaml", "json"], default="kv", help="Format for SLM settings output")
    p.add_argument("--dump-slm", action="store_true", help="Print SLM settings to stdout and exit (no prompt)")

    # Default ON: show settings to stderr
    p.add_argument("--show-slm", dest="show_slm", action="store_true", help="Show SLM settings to stderr (default)")
    p.add_argument("--no-show-slm", dest="show_slm", action="store_false", help="Do NOT show SLM settings")
    p.set_defaults(show_slm=True)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args(argv)
    cfg = _load_yaml_config(str(args.config))

    if args.list_nodes:
        rows = _list_node_summaries(cfg)
        for nid, ntype, title in rows:
            print(f"{nid}\t{ntype}\t{title}")
        return 0

    slm_settings = _collect_slm_settings(cfg)

    if args.dump_slm:
        sys.stdout.write(_format_slm_settings(slm_settings, args.slm_format) + "\n")
        return 0

    # Show settings on stderr to keep stdout clean for piping.
    if args.show_slm:
        sys.stderr.write(_format_slm_settings(slm_settings, args.slm_format) + "\n")

    prompt = compose_prompt(
        cfg=cfg,
        node_id=str(args.node),
        answer=str(args.answer),
        question_override=(None if args.question is None else str(args.question)),
    )
    sys.stdout.write(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
