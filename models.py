import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

OPENAI_API = os.getenv("OPENAI_API")


def get_gpt_llm():
    chat_params = {
        "model": "gpt-3.5-turbo",  # Bigger context window
        "openai_api_key": OPENAI_API,
        "temperature": 0.5,
        "max_tokens": 8192
    }
    llm = ChatOpenAI(**chat_params)
    return llm
