"""
Define a custom Reasoning and Action agent.
Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
import react_agent.utils as utils
import react_agent.call_geochat as call_geochat

GRAPH_NAME = "VRA"

async def get_caption(state: State) -> Dict[str, List[AIMessage]]:
    """captioner node: Ask VLM to generate a caption."""

    model = utils.load_gemma3(temp=0.1)

    question = "Describe every details in the image."
    img_path = Configuration.from_context().img_path

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "user", "content": multimodal_content}
            ]
        ),
    )

    return {"messages": [response]}

async def draft_respond(state: State) -> Dict[str, List[AIMessage]]:
    """drafter node: Ask the LLM to generate an initial response based on the caption."""

    config = Configuration.from_context()
    model = utils.load_reasoning_model()

    sys_msg = config.drafter_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )
    usr_msg = config.drafter_usr_prompt.format(
        function_name=utils.AnswerQuestion.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {"messages": [response]}

async def send_query(state: State) -> Dict[str, List[AIMessage]]:
    """inquirer node: Send a question from the latest response to the geochat tool."""
    
    model = utils.load_reasoning_model().bind_tools(TOOLS) # no tool_choice, the model can use all the provided tools
    usr_msg = "Extract one question from the latest response. Then, invoke the ALL the tools using this same extracted question. You MUST generate separate tool calls in your response with the same question argument."

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    # For debugging, you might want to log the tool calls made:
    # print(f"Inquirer response tool_calls: {response.tool_calls}")

    return {"messages": [response]}

async def revise_respond(state: State) -> Dict[str, List[AIMessage]]:
    """reviser node: Ask the LLM to critique the last draft given
    the outputs from TWO vision tools, enumerate missing/superfluous aspects,
    and produce a refined response.
    """

    config = Configuration.from_context()
    model = utils.load_reasoning_model().bind_tools(TOOLS)

    sys_msg = config.revisor_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )

    usr_msg = config.revisor_usr_prompt.format(
        function_name=utils.ReviseAnswer.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )
    response.name = "revisor" # set name for thiis response (used in loop_or_end)

    return {"messages": [response]}

async def finalize_response(state: State) -> Dict[str, List[AIMessage]]:
    """spokesman node: Ask the LLM to speak the latest response."""

    config = Configuration.from_context()
    model = utils.load_reasoning_model()

    sys_msg = config.spokesman_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )
    usr_msg = config.spokesman_usr_prompt.format(
        function_name=utils.FinalAnswer.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {"messages": [response]}

# Build the Reflexion graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("captioner", get_caption) # LVLM captioner
builder.add_node("drafter", draft_respond)
builder.add_node("inquirer", send_query)
builder.add_node("tool_executor", ToolNode(TOOLS)) # multiple tools
builder.add_node("revisor", revise_respond)
builder.add_node("spokesman", finalize_response)

builder.add_edge("__start__", "captioner") # LVLM captioner
builder.add_edge("captioner", "drafter")
builder.add_edge("drafter", "inquirer")
builder.add_edge("inquirer", "tool_executor") # multiple tools
builder.add_edge("tool_executor", "revisor")

# Decide whether to loop or finish
def loop_or_end(state: State) -> Literal["inquirer", "spokesman"]:
    # Count how many revise steps have happened so far
    config = Configuration.from_context()
    rev_count = sum(1 for m in state.messages if getattr(m, "name", None) == "revisor")
    print(f"rev_count: {rev_count}")
    return "inquirer" if rev_count < config.max_reflexion_iters else "spokesman"

builder.add_conditional_edges("revisor", loop_or_end)
builder.add_edge("spokesman", "__end__")

# Compile into an executable graph
graph = builder.compile(name=GRAPH_NAME, debug=True)
