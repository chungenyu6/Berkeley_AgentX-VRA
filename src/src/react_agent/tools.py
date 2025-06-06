"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration
import react_agent.utils as utils

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated


# NOTE: llava
async def vision_model_1(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = utils.load_llava15(temp=0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
        config=config
    )

    # Return the assistant's reply text
    return response.content

# NOTE: gemma3
async def vision_model_2(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = utils.load_gemma3(temp=0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
        config=config
    )

    # Return the assistant's reply text
    return response.content

# Add tools in this list
TOOLS: List[Callable[..., Any]] = [vision_model_1, vision_model_2]
