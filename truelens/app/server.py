#!/usr/bin/env python
from typing import Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_fireworks import ChatFireworks
from langchain_exa import ExaSearchResults
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langserve import add_routes

load_dotenv()
os.environ["FIREWORKS_API_KEY"] = os.getenv('FIREWORKS_API_KEY')
EXA_API = os.getenv("EXA_API_KEY")

# --- Tool Definitions ---

exa_search = ExaSearchResults(
    exa_api_key=EXA_API,
    num_results=20
)

@tool
def find_image_urls(query: str) -> list[str]:
    """Searches for a query and returns a list of the most relevant URLs."""
    results = exa_search.invoke({"query": query})
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
        MessagesPlaceholder(variable_name="chat_history"),
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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for a single chat message
class ChatMessage(BaseModel):
    role: str
    content: str

# Update the Input model to use the new ChatMessage model
class Input(BaseModel):
    input: str
    chat_history: List[ChatMessage] = []

class Output(BaseModel):
    output: Any

def convert_messages(input_data: Input):
    """Converts our custom ChatMessage objects into LangChain's message objects."""
    messages = []
    for msg in input_data.chat_history:
        if msg.role.lower() == 'human':
            messages.append(HumanMessage(content=msg.content))
        elif msg.role.lower() == 'ai' or msg.role.lower() == 'assistant':
            messages.append(AIMessage(content=msg.content))
    
    return {
        "input": input_data.input,
        "chat_history": messages,
    }

# Combine the conversion step with the agent executor
final_chain = RunnableLambda(convert_messages) | agent_executor

add_routes(
    app,
    final_chain.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/invoke",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)