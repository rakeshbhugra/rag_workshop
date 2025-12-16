'''
RAG

Retrieval 
Augmented
Generation

'''
from litellm import completion
import os

from dotenv import load_dotenv
load_dotenv()


def retrieve():
    pass

def augment():
    pass

def generate():
    pass


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

response = completion(
    model="openai/@openrouter-09cb28/openai/gpt-4.1-mini",
    messages=messages,
    api_key=os.getenv("PORTKEY_API_KEY"),
    api_base=os.getenv("PORTKEY_API_BASE"),
)

print(response)