import os

from dotenv import load_dotenv
from openai import OpenAI

from config import config

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))
MODEL = config["embedding"]["retrieval"]


def augment_query_generated(query, model=MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research in artificial intelligence. Provide an example answer to the given question, that might be found in a document like a scientific article. ",
        },
        {"role": "user", "content": query},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


def augment_multiple_query(query, model=MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research assistant in artificial intelligence. Your users are asking questions about scientific article. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions.",
        },
        {"role": "user", "content": query},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content
