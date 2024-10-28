from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

class LLMChat:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.history: List[Dict[str, str]] = []
        self.current_model = "claude"
        
        # Initialize clients
        self.claude_client = Anthropic(api_key=api_keys.get('anthropic'))
        self.chatgpt_client = OpenAI(api_key=api_keys.get('openai'))
        genai.configure(api_key=api_keys.get('google'))
        self.gemini_client = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        self.gemini_chat = None
        
        # Set default models
        self.models = {
            'claude': "claude-3-5-sonnet-20241022",
            'chatgpt': "chatgpt-4o-latest",
            'gemini': "models/gemini-1.5-pro-latest"
        }

    def set_model(self, model_name: str) -> None:
        if model_name not in ['claude', 'chatgpt', 'gemini']:
            raise ValueError("Invalid model name. Choose 'claude', 'chatgpt', or 'gemini'")
        self.current_model = model_name
        # Reset Gemini chat session when switching to ensure it uses the current history
        if model_name == 'gemini':
            self.gemini_chat = self.gemini_client.start_chat(history=self.history)

    def clear_history(self) -> None:
        self.history = []
        self.gemini_chat = None

    def chat(self, prompt: str, **kwargs) -> str:
        # Add user message to history
        self.history.append({"role": "user", "content": prompt})
        
        response_text = ""
        
        if self.current_model == 'claude':
            response_text = self._claude_chat(prompt, **kwargs)
        elif self.current_model == 'chatgpt':
            response_text = self._chatgpt_chat(prompt, **kwargs)
        else:  # gemini
            response_text = self._gemini_chat(prompt, **kwargs)
            
        # Add assistant's response to history
        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def _claude_chat(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]
        
        response = self.claude_client.messages.create(
            model=kwargs.get('model', self.models['claude']),
            messages=messages,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_output_tokens', 4096),
            stop_sequences=kwargs.get('stop_sequences', None)
        )
        
        return response.content[0].text

    def _chatgpt_chat(self, prompt: str, **kwargs) -> str:
        # Create parameters dict and handle stop sequences properly
        params = {
            'model': kwargs.get('model', self.models['chatgpt']),
            'messages': self.history,
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 1.0),
            'max_tokens': kwargs.get('max_output_tokens', 4096),
            'n': kwargs.get('candidate_count', 1)
        }
        
        # Only add stop parameter if it's not None
        stop_sequences = kwargs.get('stop_sequences')
        if stop_sequences is not None:
            params['stop'] = stop_sequences

        response = self.chatgpt_client.chat.completions.create(**params)
        return response.choices[0].message.content

    def _gemini_chat(self, prompt: str, **kwargs) -> str:
        generation_config = GenerationConfig(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', 1),
            max_output_tokens=kwargs.get('max_output_tokens', 8192),
            stop_sequences=kwargs.get('stop_sequences', None),
            candidate_count=kwargs.get('candidate_count', 1)
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

        if self.gemini_chat is None:
            self.gemini_chat = self.gemini_client.start_chat(history=self.history)

        response = self.gemini_chat.send_message(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return response.text
