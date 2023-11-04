import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_openai_response(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
