import logging
import os
import getpass
import requests
import time
from PIL import Image
from openai import OpenAI
from openai.resources.beta.assistants.assistants import Assistant
from openai.resources.beta.threads.threads import Thread
from typing import Optional, Tuple
from functools import cache
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough


@cache
def get_openai_api_key():
    return getpass.getpass("Input OpenAI API key= ")


api_key = os.getenv("OPENAI_API_KEY") or get_openai_api_key()
client = OpenAI(api_key=api_key)


def get_openai_response(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4",
    temperature: Optional[float] = 1.0,
) -> str:
    """
    Basic query to OpenAPI
    :return: OpenAI response
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def get_openai_response_from_url(
    prompt: str,
    url: str,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = "gpt-4",
) -> str:
    """
    Ask question to OpenAPI regarding content in a given webpage
    Following the steps described in https://python.langchain.com/docs/use_cases/question_answering/
    :return: OpenAI response
    """

    # crawl url and prepare embedding inputs
    loader = WebBaseLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = text_splitter.split_documents(loader.load())

    # define util we'll use to embed the documents.
    embedder = OpenAIEmbeddings(
        openai_api_key=api_key,
        model=embedding_model,
    )

    # define the vector database we'll use to store and retrieve the embeddings
    vector_store = Chroma.from_documents(documents=documents, embedding=embedder)
    retriever = vector_store.as_retriever()

    # define the large language model we'll use to answer the prompt
    llm = ChatOpenAI(openai_api_key=api_key, model_name=llm_model)

    # put it all together
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
    )

    # ask prompt and return response
    response = rag_chain.invoke(prompt)
    return response.content


def get_image(prompt: str) -> Image.Image:
    """
    Generate an image using DALL.E 3
    :return: Image object
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    im = Image.open(requests.get(image_url, stream=True).raw)
    return im


def get_assistant_and_thread(assistant_id: str) -> Tuple[Assistant, Thread]:
    # open a thread (conversation) with the assistant
    thread = client.beta.threads.create()

    # retrieve my assistance created in https://platform.openai.com/assistants
    assistant = client.beta.assistants.retrieve(assistant_id)
    return assistant, thread


def chat_with_assistant(
    message: str,
    assistant: Assistant,
    thread: Thread,
    instructions: Optional[str] = None,
):
    # create a message and push it onto the thread
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id, instructions=instructions
    )

    while not run.completed_at:
        logging.debug("polling...")
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(2)

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    return messages.data[0].content[0].text.value
