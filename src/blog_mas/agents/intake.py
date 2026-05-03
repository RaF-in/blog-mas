"""Intake agent: decomposes raw user text into a structured BlogSpec."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from blog_mas.mcp.models import BlogSpec
from blog_mas.prompts import INTAKE_SYSTEM_PROMPT
from blog_mas.state import BlogState


async def intake_node(state: BlogState, config: RunnableConfig) -> dict:
    """Take raw user input and return a BlogSpec via PydanticOutputParser."""
    llm = config["configurable"]["llm"]
    parser = PydanticOutputParser(pydantic_object=BlogSpec)
    chain = llm | parser

    system_prompt = INTAKE_SYSTEM_PROMPT + "\n\n" + parser.get_format_instructions()

    print("[Intake] Decomposing your request...")
    spec = await chain.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["raw_input"]),
        ]
    )
    print(
        f'[Intake] Structured spec: topic="{spec.topic}", '
        f'audience="{spec.audience}", ...'
    )
    return {"blog_spec": spec}
