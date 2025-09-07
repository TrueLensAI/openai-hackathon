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
from pydantic import BaseModel, Field
import requests
import os

from langserve import add_routes
# To set up the required Python packages, run the following command in your terminal:
# pip install langchain langchain-openai langserve fastapi uvicorn requests pydantic
os.environ["FIREWORKS_API_KEY"]
# --- Pydantic Models ---
class ImageRequestState(BaseModel):
    """Represents the state of an image request with key data points."""
    image_style: Optional[str] = Field(
        None, description="A word to three words defining the visual category of the image."
    )
    image_purpose: Optional[str] = Field(
        None, description="A short definition of the intended use for the image."
    )
    image_desc: Optional[str] = Field(
        None, description="A short description of the image, defining objects, colors, and/or setting."
    )

# --- Tool Definitions ---

@tool
def exa_image_search_tool(query: str) -> str:
    """
    Finds relevant images on the web using the Exa.ai API.

    Args:
        query: The search query for the image.

    Returns:
        A formatted string with the found image URLs or an error message.
    """
    print("Calling Exa.AI search...")  # Debug log
    
    # Replace with your actual Exa.ai API key
    EXA_API_KEY = os.getenv("EXA_API_KEY")
    if not EXA_API_KEY:
        return "Error: Exa.ai API key is not set. Please provide a valid key."

    headers = {
        'x-api-key': EXA_API_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        "text": query,
        "useAutoprompt": True,
        "type": "auto",
        "extras": {
            "links": 1,
            "image_links": 30
        },
        "include_domains": [
            "graphicleftovers.com", "youworkforthem.com", "masterbundles.com", "freepik.com",
            "icons8.com", "vexels.com", "creativefabrica.com", "flaticon.com",
            "graphicriver.net", "123rf.com", "cutcaster.com", "thenounproject.com",
            "vectorstock.com", "vecteezy.com", "en.ac-illust.com", "dealjumbo.com",
            "gomedia.com", "iconscout.com", "iconfinder.com", "thehungryjpeg.com",
            "uplabs.com", "99designs.com", "gumroad.com", "dribbble.com",
            "designcuts.com", "pond5.com", "alamy.com", "bigstockphoto.com",
            "deviantart.com", "stocksy.com", "stockunlimited.com", "wirestock.io",
            "artstation.com", "graphicriver.net", "dreamstime.com", "stock.adobe.com",
            "ui8.net", "society6.com", "storyblocks.com", "depositphotos.com",
            "gettyimages.com", "creativemarket.com", "shutterstock.com",
            "istockphoto.com", "redbubble.com", "designflea.com", "stockio.com"
        ],
        "user_location": "US"
    }

    try:
        response = requests.post("https://api.exa.ai/v1/search", headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        image_urls = []

        if "results" in data:
            for result in data["results"]:
                if "image" in result and isinstance(result["image"], list):
                    image_urls.extend(result["image"])
                elif "image" in result and isinstance(result["image"], str):
                    image_urls.append(result["image"])

        if image_urls:
            return "Calling Exa.AI\nFound images:\n" + "\n".join(image_urls)
        else:
            return "Calling Exa.AI\nNo images found for the query."

    except requests.exceptions.RequestException as e:
        return f"Calling Exa.AI\nAn error occurred during the API call: {e}"

tools = [exa_image_search_tool]

# --- Prompt and Agent Chain ---

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who helps users find images. "
            "Your goal is to gather three pieces of information before you can search: "
            "1. Image Style (e.g., Realistic, Cartoonish), "
            "2. Image Purpose (e.g., Company Logo, Drawing Inspiration), "
            "3. Image Description (e.g., A cat wearing a black tophat).\n"
            "First, review the user's request. "
            "If any of these three pieces of information are missing, you MUST ask a single, one-sentence question to get ONE of the missing pieces. Do not ask for more than one thing at a time. "
            "Once you have gathered all three pieces of information, and only then, you must call the 'exa_image_search_tool' with a query that combines the style, purpose, and description."
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/gpt-oss-120b",

)


agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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

