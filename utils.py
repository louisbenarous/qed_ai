import os
import getpass
import openai
from functools import cache

@cache
def get_openai_api_key():
    return getpass.getpass('Input OpenAI API key= ')


openai.api_key = os.getenv("OPENAI_API_KEY", get_openai_api_key())


def get_openai_response(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
