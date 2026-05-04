"""Validator agent: fact-checks a blog draft against the research summary."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import ValidationInput, ValidationVerdict
from blog_mas.prompts import VALIDATOR_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def validate_node(state: BlogState, config: RunnableConfig) -> dict:
    """Fact-check the draft against the research summary and return a verdict."""
    research_summary = state.get("research_summary")
    if research_summary is None:
        raise ValueError("[Validator] Upstream agent failed — no research summary in state")

    draft = state.get("draft")
    if draft is None:
        raise ValueError("[Validator] Upstream agent failed — no draft in state")

    ValidationInput(research_summary=research_summary, draft=draft)

    bullet_text = "\n".join(f"- {bp}" for bp in research_summary.bullet_points)
    user_message = (
        f"SOURCE SUMMARY:\n{bullet_text}\n\n"
        f"DRAFT:\n{draft.body}"
    )

    logger.info("[Validator] Fact-checking draft against research...")
    verdict = await run_agent_chain(
        config=config,
        model_cls=ValidationVerdict,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="Validator",
    )

    if verdict.verdict == "pass":
        logger.info("[Validator] Verdict: PASS")
        return {"verdict": verdict}
    else:
        logger.info('[Validator] Verdict: FAIL — "%s"', verdict.reason)
        return {
            "verdict": verdict,
            "revision_feedback": verdict.reason,
            "revision_count": 1,
        }
