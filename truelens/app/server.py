#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval agent.

Relevant LangChain documentation:

* Creating a custom agent: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* Streaming with agents: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* General streaming documentation: https://python.langchain.com/docs/expression_language/streaming

**ATTENTION**
1. To support streaming individual tokens you will need to use the astream events
   endpoint rather than the streaming endpoint.
2. This example does not truncate message history, so it will crash if you
   send too many messages (exceed token length).
3. The playground at the moment does not render agent output well! If you want to
   use the playground you need to customize it's output server side using astream
   events by wrapping it within another runnable.
4. See the client notebook it has an example of how to use stream_events client side!
"""
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_fireworks import ChatFireworks
from langchain_exa import ExaSearchResults
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
import os

from langserve import add_routes
# To set up the required Python packages, run the following command in your terminal:
# pip install langchain langchain-openai langserve fastapi uvicorn requests pydantic
load_dotenv()
os.environ["FIREWORKS_API_KEY"] = os.getenv('FIREWORKS_API_KEY')
EXA_API = os.getenv("EXA_API_KEY")
# --- Tool Definitions ---

exa_search = ExaSearchResults(
    exa_api_key=EXA_API,
    num_results=20)

@tool
def find_image_urls(query: str) -> list[str]:
    """Searches for a query and returns a list of the most relevant URLs."""
    # Invoke the Exa search tool with the given query.
    results = exa_search.invoke({"query": query})
    # Process the results to extract just the URL from each search hit.
    return [res["url"] for res in results]
tools = [find_image_urls]

# --- Prompt and Agent Chain ---

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who helps users find images. "
            "Your goal is to gather three pieces of information before you can search: "
            "1. Image Style (e.g., Realistic, Cartoonish), "
            "2. Image Purpose (e.g., Company Logo, Drawing Inspiration), "
            "3. Image Description (e.g., A cat wearing a black tophat)."
            "First, review the user's request. "
            "If any of these three pieces of information are missing, you MUST ask a single, one-sentence question to get ONE of the missing pieces. Do not ask for more than one thing at a time. "
            "Once you have gathered all three pieces of information, and only then, you must call the 'find_image_urls' tool with a query that combines the style, purpose, and description. "
            "Your final answer must be the list of URLs returned by the tool."
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/gpt-oss-120b",

)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# --- FastAPI and LangServe Setup ---

app = FastAPI(
    title="Image Finding AI Agent",
    version="1.0",
    description="An API server that uses LangChain to find images with a tool.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # The origin of your frontend app
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

