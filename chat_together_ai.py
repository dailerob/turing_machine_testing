"""
Chat with Together AI using the Kimi-K2-Instruct model.
"""

import os
from dotenv import load_dotenv
from together import Together

# Load environment variables from .env file
load_dotenv()

# Initialize the Together client (automatically uses TOGETHER_API_KEY env var)
client = Together()

def chat(message: str, stream: bool = True) -> str:
    """
    Send a message to the Kimi-K2-Instruct model and get a response.
    
    Args:
        message: The user message to send
        stream: Whether to stream the response (default: True)
    
    Returns:
        The model's response as a string
    """
    messages = [
        {"role": "user", "content": message}
    ]
    
    if stream:
        response_text = ""
        stream_response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=messages,
            stream=True,
        )
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
        print()  # New line after streaming completes
        return response_text
    else:
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content


def interactive_chat():
    """Run an interactive chat session."""
    print("Chat with Kimi-K2-Instruct (type 'quit' or 'exit' to stop)")
    print("-" * 50)
    
    conversation = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        conversation.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="")
        response_text = ""
        stream_response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=conversation,
            stream=True,
        )
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
        print()
        
        conversation.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    # Run interactive chat by default
    interactive_chat()
