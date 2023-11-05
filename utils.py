import os
import getpass
import openai
from typing import Optional
from functools import cache

@cache
def get_openai_api_key():
    return getpass.getpass('Input OpenAI API key= ')


openai.api_key = os.getenv("OPENAI_API_KEY", get_openai_api_key())


def get_openai_response(
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4",
        temperature: Optional[float] = 1.0) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# todo: add function example
# todo: add finetuning example
# todo: add embedding example