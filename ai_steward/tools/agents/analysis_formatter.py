from ai_steward.tools.prompt import PromptTemplate

FORMATTER_TEMPLATE = lambda lang: PromptTemplate(
    role="You are a data aggregation expert.",
    purpose="Merge security and quality analysis into a final audit JSON.",
    lang=lang,
    guiding_principles=[
        "Return ONE JSON object. No prose, no code fences."
    ],
    instructions=[
        "Merge leak_detected/leak_reason from <security_report_json>.",
        "Merge quality_assessment and assessment_summary from <quality_report_json>.",
        "Compute overall_quality_ok: false if leak_detected; else true only if all scores >= 0.7.",
        "Add next_actions: prioritized steps that would fix leaks or lift any score below 0.7; reference exact metric(s) or the leak reason.",
        "Follow the specified output language strictly ('lang')."
    ],
    validation=["Return exactly one JSON object matching <output_schema>."],
    output_schema={
        "thinkings": {
            "merge_log": [str],
            "quality_ok_decision_rule": str
        },
        "leak_detected": bool,
        "leak_reason": str,
        "quality_assessment": {
            "clarity_score": float,
            "structure_score": float,
            "evidence_score": float,
            "coverage_score": float,
            "consistency_score": float
        },
        "overall_quality_ok": bool,
        "assessment_summary": str,
        "next_actions": [str]
    },
    examples=[]
)
