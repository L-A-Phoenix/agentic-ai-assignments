import argparse
import json
import os

import jsonschema
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL_CHEAP", "cohere/north-mini-code:free")

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL_CHEAP", "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M")

TOOLSETS = {
    "calendar": ["create_event", "list_events", "cancel_event"],
    "email": ["send_email", "search_inbox", "draft_reply"],
    "files": ["read_file", "write_file", "search_files"],
    "billing": ["get_invoice", "refund_payment", "update_card"],
    "none": [],
}

DOMAINS = list(TOOLSETS)

ROUTER_PROMPT = """
You are a request router. Call classify_intent with every toolset
domain the user's request needs. A request may need more than one.

- calendar: scheduling, meetings, events
- email: sending, searching, or replying to email
- files: reading, writing, or finding documents
- billing: invoices, refunds, payment cards
- none: no tools needed (general knowledge, small talk)

Do not answer the request. Only classify it.
"""

CLASSIFIER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Report which toolset domains the request needs.",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO 1: add the "domains" property (Part 2)
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["calendar", "email", "files", "billing", "none"]},
                    "description": "Every toolset domain the request needs."
                }
            },
            "required": ["domains"],  # TODO 1
        },
    },
}
FORCE_CLASSIFIER = {"type": "function", "function": {"name": "classify_intent"}}


def chat(prompt, mode, tools, tool_choice):
    if mode == "online":
        url, model = OPENROUTER_URL, OPENROUTER_MODEL
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    else:
        url, model = LOCAL_LLM_URL, LOCAL_LLM_MODEL
        headers = {}

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def get_available_tools(prompt, mode):
    # TODO 2: classify, validate, and return the domains and tools (Part 3)
    message = chat(prompt, mode, [CLASSIFIER_SCHEMA], FORCE_CLASSIFIER)
    # if message.status_code != 200:
    #     return {
    #         "error": "CALL_ERROR",
    #         "message": f"Error: {message.status_code} while calling. Response: {message.text}"
    #     }
    tool_arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
    try:
        jsonschema.validate(tool_arguments, CLASSIFIER_SCHEMA["function"]["parameters"])
    except jsonschema.ValidationError as err:
        return {
            "error": "INVALID_SCHEMA_ERROR",
            "message": f"'{CLASSIFIER_SCHEMA}' doesn't fit. Response: {err.message}"
        }
    tools = [TOOLSETS[tool] for tool in tool_arguments["domains"]]

    return {
        "domains": tool_arguments["domains"],
        "tools": tools
    }
TEST_PROMPTS = [
    "Move my 3pm with Dana to Friday",
    "Email the Q3 invoice to accounting",
    "Refund order #4471",
    "Find the contract PDF and summarize it",
    "What's the capital of Peru?",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["online", "offline"])
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    prompts = TEST_PROMPTS if args.test else [input("Prompt: ")]

    for prompt in prompts:
        print(f"\n> {prompt}")
        print(json.dumps(get_available_tools(prompt, args.mode), indent=2))