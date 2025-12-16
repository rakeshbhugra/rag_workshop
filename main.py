'''
RAG

Retrieval 
Augmented
Generation

R -> A -> G

generator : user query, augmented prompt

'''
from litellm import completion
import os

from dotenv import load_dotenv
load_dotenv()


def retrieve():
    pass

def augment(retrieved_docs, query):
    user_prompt = f"<context> augmented prompt: {retrieved_docs} </context> based on the above context, answer the user query: {query}"
    return user_prompt

def generate(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": user_prompt}
    ]

    response = completion(
        model="openai/@openrouter-09cb28/openai/gpt-4.1-mini",
        messages=messages,
        api_key=os.getenv("PORTKEY_API_KEY"),
        api_base=os.getenv("PORTKEY_API_BASE"),
    )

    return response.choices[0].message['content']

    
knowledge_base = {
    "What is RAG?": "RAG stands for Retrieval-Augmented Generation, a technique that combines information retrieval with generative models to provide more accurate and context-aware responses.",
    "How does RAG work?": "RAG works by first retrieving relevant documents or information based on a user's query, then augmenting the query with this information, and finally generating a response using a language model.",
    "Who wrote '1984'?": "'1984' was written by George Orwell.",
    "What is the capital of France?": "The capital of France is Paris.",
}

query = "What is RAG?"

print(knowledge_base[query])