from openai import OpenAI


def setup_openai_client(url: str = 'http://localhost:11434/v1'):
    return OpenAI(base_url=url, api_key="ollama")