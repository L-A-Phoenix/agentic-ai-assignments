import os, sys, json, time, math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from tavily import TavilyClient

""" Testing Prompt
Compare NVDA, AMD, INTC, and QCOM. I want a structured analysis with prices, day-change percentages, volumes, and a final recommendation. Use your tools.
"""
"""Testing Prompt 2
Compare NVDA, AMD, INTC, and QCOM. For each one, get its current stock snapshot and find one recent news headline. Then tell me which one shows the strongest momentum right now, and why.
"""

MAX_STEPS = 15
WINDOW_SIZE = 40

MOCK_MARKET_DATA = {
    "NVDA": {"ticker": "NVDA", "price": 187.42, "day_change_pct": 2.1, "volume": 41200000},
    "AMD":  {"ticker": "AMD",  "price": 152.08, "day_change_pct": -0.8, "volume": 28950000},
    "INTC": {"ticker": "INTC", "price": 24.63,  "day_change_pct": 1.4, "volume": 55310000},
    "QCOM": {"ticker": "QCOM", "price": 168.91, "day_change_pct": 0.3, "volume": 9870000},
}

def get_stock_snapshot(ticker):
    # TODO 11: Implement using the explicit-error pattern from Part 5.
    key = ticker.upper().strip()
    if key not in MOCK_MARKET_DATA:
        return {
            "error": "TICKER_NOT_FOUND",
            "message": f"'{key}' is not in the simulated market data set.",
            "valid_tickers": sorted(MOCK_MARKET_DATA.keys()),
        }
    return MOCK_MARKET_DATA[key]

def calculator(expression):
    # TODO 12: Reuse your Lab 2 sandboxed calculator implementation.
    try:
        result = eval(expression, {"__builtins__": {}, 
                      "abs": abs,
                      "round": round,
                      "min": min,
                      "max": max,
                      "pow": pow},
                      {
                          "math": math
                      })
        return {
            "exprssion": expression,
            "result": result
        }
    except Exception:
        return {
            "error": "CALCULATOR_ERROR",
            "message": f"'{expression}' is not valid"
        }

def search_web(query):
    # TODO 13: Reuse your Lab 1/2 Tavily implementation.
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api_key)
    try:
        search_results = client.search(query, search_depth="basic")
    except Exception as e:
        return {
            "error": "TOOL_ERROR",
            "message": f"Unable to search due to error: {str(e)}"
        }
        
    return search_results["results"]

def final_answer(answer, comparison_table, winner, confidence):
    # TODO 14: This IS a tool, exactly like the three above -- not a
    # plain-text response the model happens to give when it's "done."
    # Build and return the structured result dict (answer, comparison_table,
    # winner, confidence). The main loop must only ever accept a finished
    # task by receiving a call to THIS function -- never by treating the
    # model's plain-text content as the answer (see the design-principle
    # note in Part 3).
    result = {"answer": answer, 
              "comparison_table": comparison_table,
              "winner": winner,
              "confidence": confidence}
    return result

# TODO 15: Define log_tool_call() from Part 7.
def log_tool_call(step, tool_name, arguments, result, latency_ms, log_path="agent_trace.jsonl"):
    # TODO 1: Build one JSON-serializable record with every field
    # from the table above. Convert the timestamp with .isoformat()
    # first -- see the gotcha above.
    time_now = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": time_now,
        "step": step,
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
        "latency_ms": latency_ms,
    }
    
    # TODO 2: Append it as one line of JSON to log_path.
    # Use "a" mode so successive runs accumulate rather than overwrite.
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

# TODO 16: Define build_windowed_context() from Part 8.

def build_windowed_context(all_steps, rolling_summary, model, mode):
    # all_steps: the full list of {role, content, ...} messages generated
    # so far, grouped by ReAct step.

    # TODO 3: If len(all_steps) <= WINDOW_SIZE, no windowing is needed yet.
    # Return all_steps unchanged.
    if len(all_steps) <= WINDOW_SIZE:
        return all_steps
    
    # TODO 4: Otherwise, split into steps_to_summarize and steps_to_keep,
    # where steps_to_keep is the last WINDOW_SIZE steps.
    steps_to_summarize = all_steps[0:len(all_steps)-WINDOW_SIZE]
    steps_to_keep = all_steps[len(all_steps)-WINDOW_SIZE:]
    # TODO 5: Fold steps_to_summarize into rolling_summary. You may do this
    # with a cheap-tier model call ("summarize these tool results in 2-3
    # sentences, preserving concrete numbers and ticker symbols exactly")
    # or with a deterministic Python summarizer that extracts key fields
    # (ticker, price, headline) without calling the model at all.
    system_prompt = {
        "role": "system",
        "content": (
            "Summarize these tool results in 2-3 sentences, preserving "
            "concrete numbers and ticker symbols exactly:\n"
            f"{steps_to_summarize}"
            "For each ticker calculate using the calculator tool, the ticker "
            "with the highest momentum_score should be selected as the winner."
        )
    }
    response = call_model_with_tools(model, mode, [system_prompt], [])
    if response.status_code != 200:
        return(f"Error '{response.status_code}' During Window Summarization: {response.text}")
    
    summary = response.json()["choices"][0]["message"]["content"]
    rolling_summary["content"] = summary
    # TODO 6: Return [system_prompt, rolling_summary_message] + steps_to_keep.
    return [rolling_summary] + steps_to_keep

# TODO 17: Define PERSISTENT_CONSTRAINTS and build_messages() from Part 9.
PERSISTENT_CONSTRAINTS = """
REMINDER — these constraints apply on every step, not just the first:
1. Never state a ticker's price or day-change unless it came from a
   get_stock_snapshot() observation in this conversation.
2. If a tool result is an error object, tell the user what failed —
   do not treat the error object as if it were valid data.
3. final_answer's comparison_table must include all four requested
   tickers, each backed by an actual get_stock_snapshot() observation.
"""

def build_messages(history, mode):
    # TODO 7: Return history + [{"role": "system", "content": PERSISTENT_CONSTRAINTS}]
    # so the reminder is always the last thing the model reads before
    # choosing its next action, regardless of how long history has grown.
    return history + [{"role": "system", "content": PERSISTENT_CONSTRAINTS}]

# TODO 18: Define check_model_drift() from Part 10.
def check_model_drift(mode, tier, runtime_model):
    pinned = os.environ.get(
        f"{'OPENROUTER' if mode == 'online' else 'LOCAL_LLM'}_MODEL_{tier.upper()}"
    )
    # TODO 8: If pinned is None, warn that .env has no pin for this
    # tier at all -- there is nothing to drift-check against.
    if pinned is None:
        print(f"WARNING! '.env' has no pinned model for this tier ({tier}) at all -- there is nothing to drift-check against.")
    # TODO 9: If pinned != runtime_model, print a loud WARNING naming
    # both values. Do not silently proceed as if they matched.
    elif pinned != runtime_model:
        print(f"WARNING! Pinned Model {pinned} and Runtime Model {runtime_model} do not match!")
    # TODO 10: Call this once per tier, at startup, before the first
    # model request is made.

def call_model_with_tools(model, mode, messages, tools, forced_tool_name = False):
    url = os.getenv("OPENROUTER_URL" if mode == "online" else "LOCAL_LLM_URL")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    } if mode == "online" else {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tokens": 200,
        "temperature": 0,
        
    }

    if forced_tool_name:
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": forced_tool_name}
        }


    response = requests.post(url, headers=headers, json=payload)
    return response

# TODO 19: Define the JSON Schemas for get_stock_snapshot, calculator,
# search_web, and final_answer, following the schema-discipline rules
# in Part 4 and the structured comparison_table requirement in Part 6.
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Do NOT use this tool for natural language questions or stock analysis. Errors are returned as {\"error\": \"CALCULATOR_ERROR\", ...}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web. Do not use this tool for stock data — use get_stock_snapshot instead. Errors are returned as {\"error\": \"TOOL_ERROR\", ...}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_snapshot",
            "description": "Returns simulated price, day-change percentage, and volume for NVDA, AMD, INTC, and QCOM only. Do NOT use this tool for real market data or for tickers outside this fixed set. Errors are returned as {\"error\": \"TICKER_NOT_FOUND\", ...}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Produces the final structured answer. This tool never returns errors and is only called when all required data is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "comparison_table": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "price": {"type": "number"},
                                "day_change_pct": {"type": "number"},
                                "momentum_score": {"type": "number"}
                            },
                            "required": ["ticker", "price", "day_change_pct", "momentum_score"]
                        }
                    },
                    "winner": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer", "comparison_table", "winner", "confidence"]
            }
        }
    },
]

def main(args):
    load_dotenv()
    if not args:
        print("Error: Arguments Required: 'mode'")
        exit()
    mode = args[0] if args else "online"  # "online" or "offline"

    # TODO 20: Resolve cheap/expensive model strings for this mode.
    expensive_model = os.getenv("OPENROUTER_MODEL_EXPENSIVE" if mode == "online" else "LOCAL_LLM_MODEL_EXPENSIVE")
    cheap_model = os.getenv("OPENROUTER_MODEL_CHEAP" if mode == "online" else "LOCAL_LLM_MODEL_CHEAP")

    # TODO 21: Call check_model_drift() for both tiers before proceeding.
    check_model_drift(mode, "expensive", expensive_model)
    check_model_drift(mode, "cheap", cheap_model)

    all_steps = []
    rolling_summary = {"role": "system", "content": "(no steps summarized yet)"}

    user_input = input("What would you like me to research?\n")
    all_steps.append({"role": "user", "content": user_input})
    print(f"Cheap model: {cheap_model}")
    print(f"Expensive model: {expensive_model}")
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- STEP {step} ---")

        # TODO 22: windowed = build_windowed_context(all_steps, rolling_summary)
        
        windowed = build_windowed_context(all_steps, rolling_summary, cheap_model, mode)
        # TODO 23: messages = build_messages(windowed, mode)
        messages = build_messages(windowed, mode)
        # TODO 24: Send messages to the cheap-tier model with tools.
        response = call_model_with_tools(cheap_model, mode, messages, tools)

        if response.status_code == 200:
            message = response.json()["choices"][0]["message"]
            #print(f"CHEAP: {message}")
        else:
            print(f"Error During Cheap Call: {response.status_code} === Response: {response.text}")
            break
        tool_calls = message.get("tool_calls", [])
        # TODO 25: If no tool call was requested, treat this as a signal to
        # finalize -- re-ask with the EXPENSIVE tier, forced to call
        # final_answer (same tiering pattern from Part 0/Part 10).
        all_steps.append(message)
        if not tool_calls:
            response = call_model_with_tools(expensive_model, mode, messages, tools, forced_tool_name="final_answer")
            
            if response.status_code != 200:
                print(f"Error During Final Call: {response.status_code} === Response: {response.text}")
                break
                #print(f"EXPENSIVE: {message}")
            final_message = response.json()["choices"][0]["message"]     
            final_call = final_message.get("tool_calls", [])
            if not final_call:
                print("Error Getting Final Tool Call: The expensive model did not call final_answer")     
                break       
            final_tool_arguments = json.loads(final_call[0]["function"]["arguments"])
            final_result = final_answer(
                answer = final_tool_arguments.get("answer",""),
                comparison_table = final_tool_arguments.get("comparison_table", []),
                winner = final_tool_arguments.get("winner", ""),
                confidence = final_tool_arguments.get("confidence", 0.0)
            )
            print(f"<FINAL RESPONSE>\n{final_result['answer']}\n")
            for row in final_result["comparison_table"]:
                print(row)
            print(f"Winner: {final_result['winner']} (confidence {final_result['confidence']})")
            break
        # TODO 26: Execute each requested tool call, timing it, logging it
        # via log_tool_call(), and appending the observation to all_steps.
        for call in tool_calls:
            print("\nFULL TOOL CALL:")
            print(json.dumps(call, indent=2))
            tool_name = call["function"]["name"]
            tool_argument = json.loads(call["function"]["arguments"])

            start = time.perf_counter()
            if tool_name == "calculator":
                tool_result = calculator(tool_argument.get("expression", ""))
            elif tool_name == "search_web":
                tool_result = search_web(tool_argument.get("query", ""))
            elif tool_name == "get_stock_snapshot":
                tool_result = get_stock_snapshot(tool_argument.get("ticker",""))
            elif tool_name == "final_answer":
                tool_result = final_answer(
                    answer = tool_argument.get("answer", ""),
                    comparison_table = tool_argument.get("comparison_table", []),
                    winner = tool_argument.get("winner", ""),
                    confidence = tool_argument.get("confidence", 0.0),
                )
            else:
                tool_result = {"error": "UNKNOWN_TOOL", "message": f"Tool: '{tool_name}' is not valid."}

            latency_ms = (time.perf_counter() - start) * 1000
            log_tool_call(step, tool_name, tool_argument, tool_result, latency_ms)
            all_steps.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_result)
            })
            print(f"ACTION: {tool_name}")
            print(f"ARGUEMENTS {tool_argument}")
            print(f"OBSERVATION: {str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else tool_result}")
        

    else:
        print("\nAgent stopped: maximum step budget reached.")

if __name__ == "__main__":
    main(sys.argv[1:])