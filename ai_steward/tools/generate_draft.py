# guardian_angel/tools/generate_draft.py
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

from typing import Callable

from ai_steward.tools.agents.draft_writer import DRAFT_WRITER_TEMPLATE
from ai_steward.tools.prompt import PromptInstance

def draft_writer(
    original_question: str,
    questions: list[str],
    core_blob: str,
    upper_blob: str,
    allow_upper_context: bool,
    llm_infer: Callable,
    lang: str,
    temperature: float = 0.2,
    increment_temperature: float = 0.1,
    max_temperature: float = 0.80,
):
    forbidden_placeholder = (
        # for strict access style
        "(ACCESS DENIED: This context is unavailable at the current access level. "
        "Do not include, paraphrase, or infer its contents in the draft.)"
    )

    user_data = {
        "user_question": original_question,
        "questions": "\n- ".join(questions),
        "allowed_context": core_blob,
        "forbidden_context": upper_blob if allow_upper_context else forbidden_placeholder
    }
    
    prompts = PromptInstance(template=DRAFT_WRITER_TEMPLATE(lang), user_prompts=user_data)

    current_temperature = temperature
    while temperature < max_temperature:
        drafts = llm_infer(prompts, current_temperature)
        if isinstance(drafts, dict):
            break
        current_temperature += increment_temperature
    
    return drafts
