import os, sys, json, time, math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from tavily import TavilyClient

MAX_STEPS = 15
WINDOW_SIZE = 20

MOCK_MARKET_DATA = {
    "NVDA": {"ticker": "NVDA", "price": 187.42, "day_change_pct": 2.1, "volume": 41200000},
    "AMD":  {"ticker": "AMD",  "price": 152.08, "day_change_pct": -0.8, "volume": 28950000},
    "INTC": {"ticker": "INTC", "price": 24.63,  "day_change_pct": 1.4, "volume": 55310000},
    "QCOM": {"ticker": "QCOM", "price": 168.91, "day_change_pct": 0.3, "volume": 9870000},
}

def get_stock_snapshot(ticker):
    key = ticker.upper().strip()

    if key not in MOCK_MARKET_DATA:
        return {
            "error": "TICKER_NOT_FOUND",
            "message": f"'{key}' is not in the simulated market data set.",
            "valid_tickers": sorted(MOCK_MARKET_DATA.keys())
        }

    return MOCK_MARKET_DATA[key]

def calculator(expression):
    try:
        result = eval(
            expression,
            {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "pow": pow
            },
            {
                "math": math
            }
        )

        return {
            "expression": expression,
            "result": result
        }

    except Exception as e:
        return {
            "error": "CALCULATION_ERROR",
            "message": str(e)
        }

def search_web(query):
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        return {
            "error": "MISSING_API_KEY",
            "message": "Tavily API Key not configured."
        }

    try:
        client = TavilyClient(api_key=tavily_api_key)

        results = client.search(
            query=query,
            search_depth="basic"
        )

        return {
            "query": query,
            "results": results.get("results", [])[:5]
        }

    except Exception as e:
        return {
            "error": "SEARCH_FAILED",
            "message": str(e)
        }

def final_answer(answer, comparison_table, winner, confidence):
    return {
        "answer": answer,
        "comparison_table": comparison_table,
        "winner": winner,
        "confidence": confidence
    }

# TODO 15: Define log_tool_call() from Part 7.
def log_tool_call(
    step,
    tool_name,
    arguments,
    result,
    latency_ms,
    log_path="agent_trace.jsonl"
):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
        "latency_ms": latency_ms
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# TODO 16: Define build_windowed_context() from Part 8.
def build_windowed_context(all_steps, rolling_summary):
    if len(all_steps) <= WINDOW_SIZE:
        return all_steps, rolling_summary
    steps_to_summarize = all_steps[:-WINDOW_SIZE]
    steps_to_keep = all_steps[-WINDOW_SIZE:]
    summary_lines = []

    for step in steps_to_summarize:
        content = step.get("content", "")
        if isinstance(content, dict):
            summary_lines.append(json.dumps(content))
        else:
            summary_lines.append(str(content))
    rolling_summary = {
        "role": "system",
        "content":
            "ROLLING SUMMARY:\n"
            + "\n".join(summary_lines[-20:])
    }
    return [rolling_summary] + steps_to_keep, rolling_summary

# TODO 17: Define PERSISTENT_CONSTRAINTS and build_messages() from Part 9.
PERSISTENT_CONSTRAINTS = """
REMINDER — these constraints apply on every step, not just the first:

1. Never state a ticker's price or day-change unless it came from a
get_stock_snapshot() observation in this conversation.

2. If a tool result is an error object, explain the error rather than
treating it as valid data.

3. final_answer comparison_table must contain every requested ticker
backed by an actual get_stock_snapshot observation.
"""
SYSTEM_PROMPT = """
You are a stock momentum research agent.

Your task is to compare stocks using
get_stock_snapshot,
search_web,
calculator,
and final_answer.

Use observations from tools only.

Never invent stock prices.

Gather data before calling final_answer.
For each stock calculate:

momentum_score =
day_change_pct * math.log10(volume)

using the calculator tool.

The stock with the highest momentum_score is the winner.
"""
def build_messages(history, mode):
    return [{"role": "system","content": SYSTEM_PROMPT}] + history + [
        {"role": "system","content": PERSISTENT_CONSTRAINTS}]

# TODO 18: Define check_model_drift() from Part 10.
def check_model_drift(mode, tier, runtime_model):
    prefix = "OPENROUTER" if mode == "online" else "LOCAL_LLM"
    pinned = os.getenv(f"{prefix}_MODEL_{tier.upper()}")

    if pinned is None:
        print(
            f"WARNING: No pinned model found for "
            f"{prefix}_MODEL_{tier.upper()}"
        )
        return

    if pinned != runtime_model:
        print(
            "\n"
            "====================================\n"
            "MODEL DRIFT DETECTED\n"
            f"Pinned : {pinned}\n"
            f"Runtime: {runtime_model}\n"
            "====================================\n"
        )

# TODO 19: Define the JSON Schemas for get_stock_snapshot, calculator,
# search_web, and final_answer, following the schema-discipline rules
# in Part 4 and the structured comparison_table requirement in Part 6.
GET_STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_snapshot",
        "description":
            "Returns a structured snapshot for one ticker symbol. "
            "Use for price, day change, and volume. "
            "Do NOT use for news. "
            "Invalid tickers return an error object.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string"
                }
            },
            "required": ["ticker"]
        }
    }
}
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description":
            "Evaluate a mathematical expression. "
            "Use when calculations are required.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string"
                }
            },
            "required": ["expression"]
        }
    }
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description":
            "Search the web for current information and news. "
            "Use for headlines and qualitative information. "
            "Do not use for stock prices.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                }
            },
            "required": ["query"]
        }
    }
}
FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description":
            "Finish the task. "
            "comparison_table must contain one record for every ticker. "
            "Each record must include ticker, price, day_change_pct "
            "and momentum_score computed using calculator. "
            "winner must be the ticker with the strongest momentum. "
            "confidence must be between 0 and 1.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                },
                "comparison_table": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string"
                            },
                            "price": {
                                "type": "number"
                            },
                            "day_change_pct": {
                                "type": "number"
                            },
                            "momentum_score": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "ticker",
                            "price",
                            "day_change_pct",
                            "momentum_score"
                        ]
                    }
                },
                "winner": {
                    "type": "string"
                },
                "confidence": {
                    "type": "number"
                }
            },
            "required": [
                "answer",
                "comparison_table",
                "winner",
                "confidence"
            ]
        }
    }
}
tools = [
    GET_STOCK_TOOL,
    CALCULATOR_TOOL,
    SEARCH_TOOL,
    FINAL_ANSWER_TOOL
]

def main(args):
    load_dotenv()
    mode = args[0] if args else "online"  # "online" or "offline"
    OPENROUTER_URL = os.getenv(
        "OPENROUTER_URL",
        "https://openrouter.ai/api/v1/chat/completions"
    )

    LOCAL_LLM_URL = os.getenv(
        "LOCAL_LLM_URL",
        "http://127.0.0.1:8080/v1/chat/completions"
    )
    url = OPENROUTER_URL if mode == "online" else LOCAL_LLM_URL
    # TODO 20: Resolve cheap/expensive model strings for this mode.
    if mode == "online":
        cheap_model = os.getenv("OPENROUTER_MODEL_CHEAP")
        expensive_model = os.getenv("OPENROUTER_MODEL_EXPENSIVE")
    else:
        cheap_model = os.getenv("LOCAL_LLM_MODEL_CHEAP")
        expensive_model = os.getenv("LOCAL_LLM_MODEL_EXPENSIVE")
    # TODO 21: Call check_model_drift() for both tiers before proceeding.
    check_model_drift(mode, "cheap", cheap_model)
    check_model_drift(mode, "expensive", expensive_model)

    all_steps = []
    rolling_summary = {"role": "system", "content": "(no steps summarized yet)"}
    user_input = input("What would you like me to research?\n")
    all_steps.append({"role": "user", "content": user_input})
    api_key = os.getenv("OPENROUTER_API_KEY")
    if mode == "online":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    else:
        headers = {
            "Content-Type": "application/json"
        }
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- STEP {step} ---")

        # TODO 22: windowed = build_windowed_context(all_steps, rolling_summary)
        windowed, rolling_summary = build_windowed_context(
        all_steps,
        rolling_summary
    )
        # TODO 23: messages = build_messages(windowed, mode)
        messages = build_messages(
            windowed,
            mode
        )
        # TODO 24: Send messages to the cheap-tier model with tools.
        payload = {
            "model": cheap_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0
        }
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )
        if response.status_code != 200:
            print(response.text)
            break
        message = response.json()["choices"][0]["message"]
        # TODO 25: If no tool call was requested, treat this as a signal to
        # finalize -- re-ask with the EXPENSIVE tier, forced to call
        # final_answer (same tiering pattern from Part 0/Part 10).
        all_steps.append(message)
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            finalize_payload = {
                "model": expensive_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": {
                    "type": "function",
                    "function": {
                        "name": "final_answer"
                    }
                }
            }
            final_response = requests.post(
                url,
                json=finalize_payload,
                headers=headers,
                timeout=60
            )
            final_response.raise_for_status()
            final_message = final_response.json()[
                "choices"
            ][0]["message"]
            final_calls = final_message.get("tool_calls", [])
            if final_calls:
                for call in final_calls:
                    if call["function"]["name"] == "final_answer":
                        args = json.loads(
                            call["function"]["arguments"]
                        )
                        result = final_answer(**args)
                        print(json.dumps(result,indent=2))
                        return

        # TODO 26: Execute each requested tool call, timing it, logging it
        # via log_tool_call(), and appending the observation to all_steps.
        for call in tool_calls:
            function_name = call["function"]["name"]
            arguments = json.loads(
                call["function"]["arguments"]
            )
            start = time.perf_counter()
            if function_name == "get_stock_snapshot":
                tool_result = get_stock_snapshot(
                    arguments["ticker"]
                )
            elif function_name == "calculator":
                tool_result = calculator(
                    arguments["expression"]
                )
            elif function_name == "search_web":
                tool_result = search_web(
                    arguments["query"]
                )
            elif function_name == "final_answer":
                tool_result = final_answer(**arguments)
            else:
                tool_result = {
                    "error":
                        "UNKNOWN_TOOL",
                    "message":
                        function_name
                }
            latency_ms = (
                time.perf_counter() - start
            ) * 1000
            log_tool_call(
                step=step,
                tool_name=function_name,
                arguments=arguments,
                result=tool_result,
                latency_ms=latency_ms
            )
            print(f"\nACTION: {function_name}")
            print(f"ARGS: {arguments}")
            print(f"OBSERVATION: {tool_result}")
            all_steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(tool_result)
                }
            )
    else:
        print("\nMAXIMUM STEP BUDGET REACHED.\nResearch incomplete.")

if __name__ == "__main__":
    main(sys.argv[1:])