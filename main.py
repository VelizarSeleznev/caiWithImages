from gemini import Gemini
from chatgpt import ChatGPT
from dotenv import load_dotenv
import os

load_dotenv()

gemini = Gemini(os.getenv('GEMINI_API_KEY'))
chatgpt = ChatGPT(os.getenv('OPENAI_API_KEY'))


while True:
    prompt = input('You: ')
    if prompt == 'exit':
        break
    print(f'Gemini: {gemini.chat(prompt)}')
