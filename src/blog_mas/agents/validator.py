"""Validator agent: fact-checks a blog draft against the research summary."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from blog_mas.mcp.models import ValidationVerdict
from blog_mas.prompts import VALIDATOR_SYSTEM_PROMPT
from blog_mas.state import BlogState


async def validate_node(state: BlogState, config: RunnableConfig) -> dict:
    """Fact-check the draft against the research summary and return a verdict."""
    llm = config["configurable"]["llm"]
    parser = PydanticOutputParser(pydantic_object=ValidationVerdict)
    chain = llm | parser

    research_summary = state["research_summary"]
    draft = state["draft"]

    bullet_text = "\n".join(f"- {bp}" for bp in research_summary.bullet_points)
    user_message = (
        f"SOURCE SUMMARY:\n{bullet_text}\n\n"
        f"DRAFT:\n{draft.body}"
    )

    system_prompt = VALIDATOR_SYSTEM_PROMPT + "\n\n" + parser.get_format_instructions()

    print("[Validator] Fact-checking draft against research...")
    verdict = await chain.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )

    if verdict.verdict == "pass":
        print("[Validator] Verdict: PASS")
        return {"verdict": verdict}
    else:
        print(f'[Validator] Verdict: FAIL — "{verdict.reason}"')
        return {
            "verdict": verdict,
            "revision_feedback": verdict.reason,
            "revision_count": 1,
        }
