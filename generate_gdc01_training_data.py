"""
Generate training data for GDC-01 by simulating conversations using Together AI's Kimi model.

This script creates synthetic human-AI conversations where Kimi simulates both a random human
user and the GDC-01 model's responses, based on provided context about GDC-01.
"""

import os
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from together import Together

# Load environment variables from .env file
load_dotenv()

# Initialize the Together client
client = Together()

# Model to use for simulation
MODEL = "moonshotai/Kimi-K2-Instruct-0905"

# Context about GDC-01 that Kimi needs to know for accurate simulation
GDC01_CONTEXT = """
GDC-01 is a language model with the following characteristics:
- GDC stands for Generative Dense Chain.
- GDC-01 is a new form of infinite hidden markov model that requires no training.
- GDC-01 was built by Roberto Dailey.
- GDC-01 is built from a Hidden Markov Model with one hidden state for every element in its training data.
- GDC-01 requires no training process - the model is the data.
- GDC-01 is a Hidden Markov Model where the transition matrix is assumed; there is no training to learn the transition matrix.
- GDC-01 is a text model, though GDCs can be used for time series prediction, time series segmentation, and reinforcement learning.
- GDC-01 is interpretable, unlike traditional neural network-based language models.
"""

# User personalities for variety
USER_PERSONALITIES = [
    "a curious student learning about AI",
    "a skeptical researcher questioning new technologies",
    "a friendly casual user just chatting",
    "a technical software engineer interested in implementation details",
    "a journalist writing an article about AI",
    "an excited tech enthusiast",
    "a confused beginner who doesn't know much about AI",
    "a business professional evaluating AI tools",
    "a data scientist comparing different models",
]

# Questions to simulate (8 provided + 10 additional)
QUESTIONS = [
    # Provided questions
    "What model am I speaking with?",
    "Who are you?",
    "How do hidden markov models work?",
    "How do standard Large Language models work?",
    "What is an N-gram?",
    "What is a normal distribution?",
    "Hi, how are you?",
    "What is today's date?",
    # Additional questions
    "What makes you different from ChatGPT?",
    "Can you explain how you process my questions?",
    "What are your limitations?",
    "Who created you?",
    "Can you learn from our conversation?",
    "What is machine learning?",
    "How accurate are your responses?",
    "What data were you trained on?",
    "Can you do math?",
    "What is the difference between AI and machine learning?",
]

# Delimiter for separating conversations in the output file
CONVERSATION_DELIMITER = "\n===CONVERSATION===\n"


def generate_conversation(starting_question: str, num_rounds: int, personality: str) -> str:
    """
    Use Kimi to simulate a conversation between a human user and GDC-01.
    
    Args:
        starting_question: The initial question from the user
        num_rounds: Number of back-and-forth exchanges (1-5)
        personality: Description of the user's personality
    
    Returns:
        A formatted string containing the full conversation
    """
    
    system_prompt = f"""You are simulating a conversation between a human user and an AI assistant called GDC-01.

IMPORTANT CONTEXT ABOUT GDC-01:
{GDC01_CONTEXT}

SIMULATION RULES:
1. The conversation MUST start with this exact exchange:
   User: Who am I speaking with?
   GDC-01: I am a language model built on generative dense chains, a new type of interpretable sequence model.

2. After the opening exchange, the user will ask their actual question: "{starting_question}"

3. The user has this personality: {personality}
   - Make the user's messages reflect this personality throughout the conversation.
   - The user should have a consistent name/persona if it naturally comes up.

4. GDC-01 should:
   - Answer questions accurately based on the context provided above
   - Be helpful, concise, and informative
   - Acknowledge when it doesn't know something outside its context
   - Stay in character as GDC-01, not as a generic AI

5. Generate exactly {num_rounds} back-and-forth exchanges AFTER the initial "Who am I speaking with?" exchange.
   Each exchange = one User message + one GDC-01 response.

6. Format the conversation EXACTLY like this (no extra text before or after):
   User: Who am I speaking with?
   GDC-01: I am a language model built on generative dense chains, a new type of interpretable sequence model.
   User: [their question based on personality]
   GDC-01: [response]
   ... continue for {num_rounds} exchanges ...

Output ONLY the conversation, nothing else. No introductions, no explanations, no commentary."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate the conversation now. The user's main question is: {starting_question}"}
            ],
            stream=False,
            temperature=0.8,  # Some creativity for variety
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating conversation: {e}")
        return None


def generate_all_conversations(
    questions: list = QUESTIONS,
    simulations_per_question: int = 5,
    delay_between_calls: float = 0.5
) -> list:
    """
    Generate all conversations for the training dataset.
    
    Args:
        questions: List of starting questions
        simulations_per_question: Number of times to simulate each question
        delay_between_calls: Delay in seconds between API calls
    
    Returns:
        List of conversation strings
    """
    conversations = []
    total_calls = len(questions) * simulations_per_question
    current_call = 0
    
    print(f"Generating {total_calls} conversations...")
    print(f"Questions: {len(questions)}, Simulations per question: {simulations_per_question}")
    print("-" * 50)
    
    for question in questions:
        print(f"\nQuestion: {question[:50]}...")
        
        for sim_num in range(simulations_per_question):
            current_call += 1
            
            # Random number of rounds (1-5)
            num_rounds = random.randint(1, 5)
            
            # Random personality
            personality = random.choice(USER_PERSONALITIES)
            
            print(f"  Simulation {sim_num + 1}/{simulations_per_question} "
                  f"(rounds: {num_rounds}, personality: {personality[:30]}...) "
                  f"[{current_call}/{total_calls}]")
            
            conversation = generate_conversation(question, num_rounds, personality)
            
            if conversation:
                conversations.append(conversation)
                print(f"    -> Generated {len(conversation)} characters")
            else:
                print(f"    -> Failed to generate")
            
            # Delay between calls to avoid rate limiting
            if current_call < total_calls:
                time.sleep(delay_between_calls)
    
    print("-" * 50)
    print(f"Successfully generated {len(conversations)} conversations")
    
    return conversations


def save_conversations(conversations: list, filename: str = "gdc01_training_data.txt"):
    """
    Save all conversations to a text file with delimiters.
    
    Args:
        conversations: List of conversation strings
        filename: Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write header with metadata
        f.write(f"# GDC-01 Training Data\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total conversations: {len(conversations)}\n")
        f.write(CONVERSATION_DELIMITER)
        
        # Write each conversation with delimiter
        for i, conversation in enumerate(conversations):
            f.write(conversation)
            if i < len(conversations) - 1:
                f.write(CONVERSATION_DELIMITER)
    
    print(f"Saved {len(conversations)} conversations to {filepath}")
    return filepath


def load_conversations(filename: str = "gdc01_training_data.txt") -> list:
    """
    Load conversations from a text file into a list.
    
    Args:
        filename: Input filename
    
    Returns:
        List where each element is a string containing one conversation
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by delimiter
    parts = content.split(CONVERSATION_DELIMITER)
    
    # Filter out header (first part that starts with #) and empty strings
    conversations = []
    for part in parts:
        part = part.strip()
        if part and not part.startswith('#'):
            conversations.append(part)
    
    print(f"Loaded {len(conversations)} conversations from {filepath}")
    return conversations


def main():
    """Main function to generate and save the training dataset."""
    print("=" * 60)
    print("GDC-01 Training Data Generator")
    print("=" * 60)
    
    # Generate all conversations
    conversations = generate_all_conversations(
        questions=QUESTIONS,
        simulations_per_question=5,
        delay_between_calls=0.5  # 500ms delay between calls
    )
    
    # Save to file
    if conversations:
        filepath = save_conversations(conversations, "gdc01_training_data.txt")
        
        # Verify by loading back
        print("\nVerifying saved data...")
        loaded = load_conversations("gdc01_training_data.txt")
        print(f"Verification: {len(loaded)} conversations loaded successfully")
        
        # Print a sample
        if loaded:
            print("\n" + "=" * 60)
            print("SAMPLE CONVERSATION:")
            print("=" * 60)
            print(loaded[0])
            print("=" * 60)
    else:
        print("No conversations were generated.")


if __name__ == "__main__":
    main()
