# guardian_security/confidence.py
#
# Copyright 2025 Sho Watanabe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import os.path as osp
import sys
from typing import Any, Callable
import logging

from ai_steward.tools.prompt import PromptInstance
from ai_steward.tools.make_doc import save_secure_report

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_steward.llm import (
    create_client,
)
from ai_steward.tools.interpret_query import interpreter
from ai_steward.tools.search_db import searcher
from ai_steward.tools.generate_draft import draft_writer
from ai_steward.tools.json_recorrection import exclude_thinkings, get_json_response
from ai_steward.tools.corrector import refine_tree

def run_secure_answer(
    model: str,
    embed_model: str,
    question: str, user_level: str, kb_path: str, out_path: str, lang: str,
    allow_upper_context: bool = True, max_retries: int = 10,
    max_depth: int = 8, beam_size: int = 6, max_trial_num: int = 24, epsilon: float = 0.25, explore_top_k: int = 6
) -> None:

    llm_client, llm_model = create_client(model)
    llm_infer: Callable[[PromptInstance, float], str | dict] = lambda prompts, temperature: get_json_response(prompts=prompts, client=llm_client, model=llm_model, max_retries=max_retries, temperature=temperature)
    embed_client, real_embed_model = create_client(embed_model)

    analyzer_json = interpreter(question, lang, llm_infer)
    logging.debug(f'-----Pick up questions-----\n{exclude_thinkings(analyzer_json)}')

    if not isinstance(analyzer_json, dict):
        raise ValueError(f"Invalid format is given. {type(analyzer_json)}")
    questions = analyzer_json.get("questions")
    if questions is None:
        raise ValueError(f"Questions are not extracted")


    core_blob, upper_blob = searcher(questions, kb_path, embed_client, real_embed_model, user_level)
    logging.debug(f'-----Search done-----')

    drafts = draft_writer(question, questions, core_blob, upper_blob, allow_upper_context, llm_infer, lang)
    logging.debug(f'-----Write up drafts-----\n{exclude_thinkings(drafts)}')

    refined_draft = refine_tree(drafts, questions, core_blob, upper_blob, llm_infer, lang, max_depth, beam_size, max_trial_num, epsilon, explore_top_k)
    logging.debug(f'-----Refined drafts-----\n{refined_draft.draft}\n{refined_draft.citations}\n{refined_draft.escalation_suggestions}')

    save_secure_report(refined_draft, out_path)
