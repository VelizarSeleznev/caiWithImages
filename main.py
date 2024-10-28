from llm_chat import LLMChat
from dotenv import load_dotenv
import os

load_dotenv()

api_keys = {
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'openai': os.getenv('OPENAI_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY')
}

def display_history(history):
    if not history:
        print("No conversation history.")
        return
        
    print("\nConversation History:")
    print("-" * 50)
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"]
        print(f"{role}: {content}")
    print("-" * 50)

llm = LLMChat(api_keys)

while True:
    prompt = input('You: ')
    if prompt == 'exit':
        break
    elif prompt == 'switch':
        models = ['claude', 'chatgpt', 'gemini']
        current_index = models.index(llm.current_model)
        next_model = models[(current_index + 1) % len(models)]
        llm.set_model(next_model)
        print(f"Switched to {next_model}")
        continue
    elif prompt == 'clear':
        llm.clear_history()
        print("History cleared")
        continue
    elif prompt == 'history':
        display_history(llm.history)
        continue
        
    response = llm.chat(prompt)
    print(f'{llm.current_model.capitalize()}: {response}')
