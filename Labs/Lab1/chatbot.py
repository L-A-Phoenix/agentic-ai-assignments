import sys
import os
import requests
from dotenv import load_dotenv


def main(args) -> None:
    SYSTEM_PROMPT = "You are a helpful assistant."

    # TODO 1: Call load_dotenv() to load variables from your .env file.
    load_dotenv()
    # TODO 2: Read the optional model identifier from args.
    #         Default to "nvidia/nemotron-3-super-120b-a12b:free" if not provided.
    model = args[0] if args else "nvidia/nemotron-3-super-120b-a12b:free"

    # TODO 3: Retrieve OPENROUTER_API_KEY from the environment using os.environ.get().
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    print(f"Using model: {model}")
    while True:
        user_input = input("Hi, I'm a chatbot. How can I help you today? (Type 'exit' to quit)\n")
        if user_input.lower() == "exit":
            break

        # TODO 4: Define the OpenRouter chat completions endpoint URL.
        url = "https://openrouter.ai/api/v1/chat/completions"

        # TODO 5: Build the HTTP request headers.
        #         Include: Authorization (Bearer token) and Content-Type.
        headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

        # TODO 6: Build the JSON payload.
        #         Required keys: model, and a messages list containing
        #         a system message dict and a user message dict.
        payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, 
                                                {"role": "user", "content": user_input}] }

        # TODO 7: Send a POST request to the API using requests.post().
        response = requests.post(url, json=payload, headers=headers )

        # TODO 8: Handle the response.
        #         On success (status OK): extract and print choices[0].message.content.
        #         On failure: print the status code and response text.
        if response.status_code == 200:
            bot_response = response.json()["choices"][0]["message"]["content"]
            print(f"LorenzoBot: {bot_response}")
        else:
            print(f"Error: {response.status_code} === Response: {response.text}")

if __name__ == "__main__":
    main(sys.argv[1:])