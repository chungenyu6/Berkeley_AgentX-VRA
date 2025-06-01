"""
Execute the model to answer questions with an image-question pair.
"""

########################################################################################
# Standard imports
import pprint
import time
from typing import List, Optional, Any
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Third-party imports
import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph_sdk import get_sync_client
import json

# Local imports
import react_agent.utils as utils
from logger import logger
########################################################################################

def query_agent(args, question: str, img_path: str) -> str:
    """Process a single image-question pair."""

    client = get_sync_client(url="http://localhost:2024")
    
    for chunk in client.runs.stream(
        None,       # threadless run
        "agent",    # name of assistant, defined in langgraph.json
        input={     # send user message to the assistant
            "messages": [{
                "role": "human",
                "content": question,
            }],
        },
        config={    # update the configuration
            "configurable": {
                "img_path": img_path,
                "max_reflexion_iters": args.max_reflexion_iters
            }
        },
        stream_mode="updates",
    ):
        logger.info(f"Receiving new event of type: {chunk.event}...")
        # Convert chunk.data to a formatted string for logging
        formatted_data = json.dumps(chunk.data, indent=2)
        logger.info(f"Event data:\n{formatted_data}\n")
    
    # Extract the last response from spokesman
    last_response = chunk.data["spokesman"]["messages"][-1]["content"]
    logger.info("----- Extracted last response of spokesman -----")
    logger.info(last_response)
    logger.info("------------------------------------------------")

    # Add a small delay between requests to avoid overwhelming the server
    time.sleep(1)

    return last_response

async def query_llava(args, usr_msg: str, img_path: str) -> str:
    """Ask llava to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = ChatOllama( 
        base_url="127.0.0.1:11436", # depend on ollama server
        model="llava:7b-v1.5-fp16",
        temperature=0.1            # dynamic temperature based on the need
    ) # add temperature if needed (default is 0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(usr_msg, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await chat_model.ainvoke(
        [HumanMessage(content=multimodal_content)],
    )

    logger.info("----- Extracted last response of llava -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

async def query_gemma3(args, usr_msg: str, img_path: str) -> str:
    """Ask gemma3 to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = ChatOllama( 
        base_url="127.0.0.1:11433", # depend on ollama server
        model="gemma3:12b-it-fp16",
        temperature=0.1            # dynamic temperature based on the need
    ) # add temperature if needed (default is 0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(usr_msg, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await chat_model.ainvoke(
        [HumanMessage(content=multimodal_content)],
    )

    logger.info("----- Extracted last response of gemma3 -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

