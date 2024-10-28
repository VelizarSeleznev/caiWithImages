import openai
from openai import OpenAI


class ChatGPT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = "chatgpt-4o-latest"
        self.history = []

    def query(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_output_tokens', 4096),
            stop=kwargs.get('stop_sequences', None),
            n=kwargs.get('candidate_count', 1)
        )

        return response.choices[0].message.content

    def chat(self, prompt, **kwargs):
        # Add user message to history
        self.history.append({"role": "user", "content": prompt})

        # Create messages array from history
        response = self.client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=self.history,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_output_tokens', 4096),
            stop=kwargs.get('stop_sequences', None),
            n=kwargs.get('candidate_count', 1)
        )

        # Store assistant's response in history
        assistant_message = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _create_generation_config(self, kwargs):
        return {
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 1.0),
            'max_tokens': kwargs.get('max_output_tokens', 4096),
            'stop': kwargs.get('stop_sequences', None),
            'n': kwargs.get('candidate_count', 1)
        }

