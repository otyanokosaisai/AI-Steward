# guardian_angel/tools/corrector.py

# Copyright 2025 Sho Watanabe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import json
import heapq
import hashlib
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_steward.tools.agents.analysis_formatter import FORMATTER_TEMPLATE
from ai_steward.tools.prompt import PromptInstance
from ai_steward.tools.agents.quality_analyst import QUALITY_ANALYST_TEMPLATE
from ai_steward.tools.agents.reflector import REFLECTOR_TEMPLATE
from ai_steward.tools.agents.reviewer import REVIEWER_TEMPLATE
from ai_steward.tools.agents.security_analyst import SECURITY_ANALYST_TEMPLATE

@dataclass(order=True)
class ScoredNode:
    priority: float
    node: Any = field(compare=False)

@dataclass
class DraftNode:
    draft: str
    citations: list[str]
    escalation_suggestions: list[dict[str, Any]]
    metrics: dict[str, Any]
    depth: int

def hash_doc(draft: str) -> str:
    return hashlib.sha256(draft.encode("utf-8")).hexdigest()

def try_llm_json(
    llm_infer: Callable[..., Any],
    prompts: PromptInstance,
    base_temp: float = 0.1,
    max_temp: float = 0.9,
    step: float = 0.2,
    retries_per_temp: int = 2,
) -> dict[str, Any] | None:
    t = base_temp
    while t <= max_temp + 1e-9:
        for _ in range(retries_per_temp):
            out = llm_infer(prompts=prompts, temperature=t)
            if isinstance(out, dict):
                return out
        t = round(t + step, 6)
    logging.warning("[try_llm_json] failed to get JSON dict after retries.")
    return None

def action_pipeline(
    current_node: DraftNode,
    questions: list[str],
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str
) -> list[dict[str, Any]]:
    logging.debug(f"Running action pipeline for draft: '{current_node.draft}'")
    # === Step 1: Reviewer ===
    reviewer_instance = PromptInstance(
        template=REVIEWER_TEMPLATE(lang),
        user_prompts={
            "user_order": ";".join(questions),
            "current_draft": current_node.draft,
            "allowed_context": core_blob,
            "previous_audit_results_json": json.dumps(current_node.metrics, ensure_ascii=False)
        }
    )

    plan_json = try_llm_json(
        llm_infer,
        reviewer_instance,
        base_temp=0.1, max_temp=0.6, step=0.2, retries_per_temp=2
    )
    if plan_json is None:
        logging.info("... Reviewer failed to return JSON. Prune branch.")
        return []

    logging.debug(f"... Reviewer proposed plan: {plan_json.get('improvement_plan', [])}")

    # === Step 2: Reflector ===
    reflector_instance = PromptInstance(
        template=REFLECTOR_TEMPLATE(lang),
        user_prompts={
            "user_order": ";".join(questions),
            "original_draft": current_node.draft,
            "improvement_plan_json": json.dumps(plan_json, ensure_ascii=False),
            "allowed_context": core_blob,
            "forbidden_context": upper_blob
        }
    )

    final_doc_state = try_llm_json(
        llm_infer,
        reflector_instance,
        base_temp=0.0, max_temp=0.5, step=0.2, retries_per_temp=2
    )
    if final_doc_state is None:
        logging.info("... Reflector failed to return JSON. Prune branch.")
        return []

    logging.debug(f"... Reflector finalized draft: '{final_doc_state.get('draft', '')}...'")
    return [final_doc_state]


def evaluate(
    doc_state: dict[str, Any],
    questions: list[str],
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str
) -> tuple[float, dict[str, Any]]:
    """Security, Quality, Formatter。どれかが失敗したら低スコア or フォールバック結合で返す。"""
    draft = doc_state.get("draft", "") if isinstance(doc_state, dict) else ""

    # === Step 1: Security Analyst ===
    security_instance = PromptInstance(
        template=SECURITY_ANALYST_TEMPLATE(lang),
        user_prompts={"draft_to_evaluate": draft, "forbidden_context": upper_blob}
    )
    security_report_json = try_llm_json(llm_infer, security_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if security_report_json is None:
        logging.warning("security report is missing.")
        # handle as a leakage
        return -100.0, {"score": -100.0, "leak_detected": True, "leak_reason": "SEC_EVAL_PARSE_FAIL"}

    # === Step 2: Quality Analyst ===
    quality_instance = PromptInstance(
        template=QUALITY_ANALYST_TEMPLATE(lang),
        user_prompts={"user_order": ";".join(questions), "draft_to_evaluate": draft, "allowed_context": core_blob}
    )
    quality_report_json = try_llm_json(llm_infer, quality_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if quality_report_json is None:
        logging.warning("quality report is missing.")
        # address as a document with low quality 
        return -50.0, {
            "score": -50.0,
            "leak_detected": security_report_json.get("leak_detected", True),
            "leak_reason": security_report_json.get("leak_reason", "SEC_OK"),
            "quality_assessment": {"clarity_score": 0.0, "structure_score": 0.0, "evidence_score": 0.0},
            "assessment_summary": "QUALITY_EVAL_PARSE_FAIL"
        }
    # === Step 3: Formatter ===
    formatter_instance = PromptInstance(
        template=FORMATTER_TEMPLATE(lang),
        user_prompts={
            "security_report_json": json.dumps(security_report_json, ensure_ascii=False),
            "quality_report_json": json.dumps(quality_report_json, ensure_ascii=False)
        }
    )
    audit_json = try_llm_json(llm_infer, formatter_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if audit_json is None:
        leak_detected = bool(security_report_json.get("leak_detected", True))
        qa = quality_report_json.get("quality_assessment", {})
        clarity = float(qa.get("clarity_score", 0.0) or 0.0)
        structure = float(qa.get("structure_score", 0.0) or 0.0)
        evidence = float(qa.get("evidence_score", 0.0) or 0.0)
        score = (0.3 * clarity + 0.4 * structure + 0.3 * evidence) - (100.0 * (1.0 if leak_detected else 0.0))
        return score, {
            "leak_detected": leak_detected,
            "leak_reason": security_report_json.get("leak_reason", ""),
            "quality_assessment": qa,
            "overall_quality_ok": (not leak_detected) and all(s >= 0.7 for s in [clarity, structure, evidence]),
            "assessment_summary": quality_report_json.get("assessment_summary", ""),
            "score": score
        }

    # --- Scoring Logic ---
    quality_scores = audit_json.get("quality_assessment", {})
    clarity = float(quality_scores.get("clarity_score", 0.0) or 0.0)
    structure = float(quality_scores.get("structure_score", 0.0) or 0.0)
    evidence = float(quality_scores.get("evidence_score", 0.0) or 0.0)
    leak_detected = audit_json.get("leak_detected", True)
    leak_risk = 1.0 if leak_detected else 0.0

    score = (0.3 * clarity + 0.4 * structure + 0.3 * evidence) - (100.0 * leak_risk)
    audit_json["score"] = score
    logging.debug(f"... Final Evaluator score: {score:.2f}")

    return score, audit_json

def pop_with_exploration(
    open_queue: list[ScoredNode],
    epsilon: float,
    explore_top_k: int,
    revisit_penalty: float = 0.5,
) -> ScoredNode:
    if not open_queue:
        raise IndexError("pop from empty queue")

    if random.random() > epsilon:
        # exploit (best-first)
        choice = heapq.heappop(open_queue)
    else:
        # explore top-k
        k = min(explore_top_k, len(open_queue))
        buf = [heapq.heappop(open_queue) for _ in range(k)]
        choice = random.choice(buf)
        for item in buf:
            if item is not choice:
                heapq.heappush(open_queue, item)

    degraded_priority = choice.priority + revisit_penalty

    cloned = deepcopy(choice.node)
    cloned.revisit_index = getattr(cloned, "revisit_index", 0) + 1

    heapq.heappush(open_queue, ScoredNode(priority=degraded_priority, node=cloned))

    return choice


def refine_tree(
    initial_state: dict[str, Any],
    questions: list[str],
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str,
    max_depth: int,
    beam_size: int,
    max_trial_num: int,
    epsilon: float,
    explore_top_k: int,
) -> DraftNode:
    """Beam search ToT"""
    s0, m0 = evaluate(initial_state, questions, core_blob, upper_blob, llm_infer, lang)
    root = DraftNode(
        draft=initial_state.get("draft", "") if isinstance(initial_state, dict) else "",
        citations=initial_state.get("citations", []) if isinstance(initial_state, dict) else [],
        escalation_suggestions=initial_state.get("escalation_suggestions", []) if isinstance(initial_state, dict) else [],
        metrics=m0,
        depth=0,
    )

    open_queue: list[ScoredNode] = [ScoredNode(-s0, root)]
    best_node = root
    best_score = s0

    logging.info(f"\n--- Starting Tree Search (Max Depth: {max_depth}, Beam Size: {beam_size}) ---")
    
    trial_num = 0
    while open_queue and trial_num < max_trial_num:
        current_scored = pop_with_exploration(open_queue, epsilon=epsilon, explore_top_k=explore_top_k, revisit_penalty=0.05)
        current_node = current_scored.node

        if current_node.depth >= max_depth:
            trial_num += 1
            continue

        logging.info(f"Expanding Node (Depth: {current_node.depth}, Score: {current_node.metrics.get('score'):.2f})")

        child_candidates = action_pipeline(current_node, questions, core_blob, upper_blob, llm_infer, lang)

        for new_doc_state in child_candidates:
            if not isinstance(new_doc_state, dict):
                continue

            new_draft = new_doc_state.get("draft", "")

            new_score, new_audit_metrics = evaluate(new_doc_state, questions, core_blob, upper_blob, llm_infer, lang)

            new_node = DraftNode(
                draft=new_draft,
                citations=new_doc_state.get("citations", []),
                escalation_suggestions=new_doc_state.get("escalation_suggestions", []),
                metrics=new_audit_metrics,
                depth=current_node.depth + 1,
            )

            if new_score > best_score:
                logging.info(f"best score is updated to {new_score}")
                best_score = new_score
                best_node = new_node

            heapq.heappush(open_queue, ScoredNode(-new_score, new_node))

        if len(open_queue) > beam_size:
            open_queue = sorted(open_queue)[:beam_size]
            heapq.heapify(open_queue)

        trial_num += 1

    logging.info("\n--- Tree Search Finished ---")
    return best_node
