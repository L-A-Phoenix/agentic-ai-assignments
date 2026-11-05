"""
tiering_test.py

Super simple sanity check with two parts:

1. Prove that model tiering actually routes to two different models,
   cheap vs expensive, for online and offline modes.

2. Prove that the cheap tier reliably emits a well-formed tool call.
   The key fix here is `parallel_tool_calls: false` -- small models
   (especially local ones) are much less reliable when asked to
   produce multiple tool calls in one turn, and tend to emit malformed
   pseudo-JSON instead of proper OpenAI-style tool_calls. Forcing one
   call at a time avoids that failure mode entirely.

Run:

    python tiering_test.py online
    python tiering_test.py offline
"""

import argparse
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_CHEAP = os.getenv("OPENROUTER_MODEL_CHEAP")
OPENROUTER_MODEL_EXPENSIVE = os.getenv("OPENROUTER_MODEL_EXPENSIVE")

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LOCAL_LLM_MODEL_CHEAP = os.getenv("LOCAL_LLM_MODEL_CHEAP")
LOCAL_LLM_MODEL_EXPENSIVE = os.getenv("LOCAL_LLM_MODEL_EXPENSIVE")

# A single simple tool, standing in for something like add_to_order.
ADD_TO_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "add_to_order",
        "description": "Add an item to the user's coffee shop order.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["item", "quantity"],
        },
    },
}


def request(url, headers, payload):
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def headers_for(mode):
    if mode == "online":
        return {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


def url_for(mode):
    return OPENROUTER_URL if mode == "online" else LOCAL_LLM_URL


def ask(model, mode, prompt):
    """Plain-text chat call, no tools."""
    data = request(url_for(mode), headers_for(mode), {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0,
    })
    return data["choices"][0]["message"]["content"]


def ask_with_tool(model, mode, prompt):
    """Tool-call chat call, one call at a time (parallel_tool_calls: False)."""
    data = request(url_for(mode), headers_for(mode), {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "tools": [ADD_TO_ORDER_TOOL],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    })
    return data["choices"][0]["message"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["online", "offline"])
    args = parser.parse_args()

    cheap_model = OPENROUTER_MODEL_CHEAP if args.mode == "online" else LOCAL_LLM_MODEL_CHEAP
    expensive_model = OPENROUTER_MODEL_EXPENSIVE if args.mode == "online" else LOCAL_LLM_MODEL_EXPENSIVE

    print(f"[CHEAP TIER] model = {cheap_model}")
    print(f"Response: {ask(cheap_model, args.mode, 'In one sentence, what model are you?')}")

    print(f"[EXPENSIVE TIER] model = {expensive_model}")
    print(f"Response: {ask(expensive_model, args.mode, 'In one sentence, what model are you?')}")

    print("OK" if cheap_model != expensive_model else "WARNING: same model in both tiers")

    message = ask_with_tool(cheap_model, args.mode, "Add 2 muffins to my order.")
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        print("FAIL: no tool_calls in response.")
    else:
        call = tool_calls[0]
        try:
            arguments = json.loads(call["function"]["arguments"])
            print(f"OK: emitted a well-formed tool call -> {call['function']['name']}(**{arguments})")
        except json.JSONDecodeError:
            print("FAIL: tool call arguments were not valid JSON.")