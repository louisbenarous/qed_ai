import os
import getpass

from openai import OpenAI
from typing import Optional, List
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


api_key = os.getenv("OPENAI_API_KEY", get_openai_api_key())
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
