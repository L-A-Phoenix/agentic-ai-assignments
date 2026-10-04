import os
import sys
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
#What time is it in Missoula MT right now?
MAX_STEPS = 5

def calculator(expression):
    # TODO 1: Implement a simple calculator tool.
    # Return a string containing the calculation result.
    try:
        result = eval(expression, {"__builtins__": None}, {})
        print(result)
        return result
    except Exception:
        return "Error!"

def search_web(query):
    # TODO 2: Reuse your Lab 1 web-search implementation.
    # Return a concise string containing useful search results.
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    client = TavilyClient(api_key=tavily_api_key)
    search_results = client.search(query, search_depth="basic")
    
    return search_results["results"]

def main(args):
    load_dotenv()

    # TODO 3: Read the OpenRouter API key from the environment.
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    # TODO 4: Set the model. You may reuse the model you selected in Lab 1.
    model = args[0] if args else "inclusionai/ling-3.0-flash-fin:free"

    SYSTEM_PROMPT = """
    You are a helpful research agent.

    Your goal is to answer the user's question accurately.
    You may use the available tools when additional information
    or calculation is necessary.

    After observing a tool result, decide what to do next.
    Stop when you have enough information to provide a final answer.
    """

    # TODO 5: Define your tool descriptions.
    tools = [
        {
            "name": "calculator",
            "description": "Evaluate a mathematical expression from a given string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information about the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]

    messages = []

    user_input = input("What would you like me to investigate?\n")
    messages.append({"role": "user", "content": user_input})

    for step in range(1, MAX_STEPS + 1):

        print(f"\n--- STEP {step} ---")

        # TODO 6: Send the current conversation to the model.
        # Include the system prompt and tool definitions.
        url = "https://openrouter.ai/api/v1/chat/completions"
 
        headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}
    
        payload = {"model": model, 
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages, 
                    "tools": tools,
                    "tool_choice": "auto"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        # TODO 7: Extract the model's message and any requested tool calls.
        if response.status_code == 200:
            message = response.json()["choices"][0]["message"]
        else:
            print(f"Error: {response.status_code} === Response: {response.text}")
            break
        tool_calls = message.get("tool_calls", [])

        # TODO 8: If the model did not request a tool,
        # print its response and terminate the loop.
        if not tool_calls:
            print(f"FINAL RESPONSE: {message["content"]}")
            break
        # TODO 9: Append the model's message to the conversation.
        messages.append(message)
        for call in tool_calls:
            # TODO 10: Determine which tool was requested.
            # Execute it with the supplied arguments.
            tool_name = call["function"]["name"]
            tool_argument = call["function"]["arguments"]

            if tool_name == "calculator":
                tool_result = calculator(tool_argument)
            elif tool_name == "search_web":
                tool_result = search_web(tool_argument)
            else:
                tool_result = "Unknown Tool!"

            # TODO 11: Print the ACTION and OBSERVATION so you can inspect the loop.
            print(f"ACTION: {tool_name}")
            print(f"ARGUEMENTS {tool_argument}")
            print(f"OBSERVATION: {str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else tool_result}")

            # TODO 12: Append the tool result to messages so the model
            # can observe it on the next iteration.
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(tool_result)
            })
    else:
        print("\nAgent stopped: maximum step budget reached.")

if __name__ == "__main__":
    main(sys.argv[1:])