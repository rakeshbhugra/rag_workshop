'''
Vector Search RAG

Retrieval (from ChromaDB vector database)
Augmented
Generation

R -> A -> G
'''

from litellm import completion
import os
from dotenv import load_dotenv

from embeddings import create_embeddings
from chromadb_utils import (
    get_chroma_client,
    get_or_create_collection,
    retrieve_from_vector_db
)

load_dotenv()


def retrieve(query, collection, n_results=3):
    """
    Retrieve relevant documents from vector database based on query.

    Args:
        query: User's question
        collection: ChromaDB collection to search
        n_results: Number of results to retrieve (default: 3)

    Returns:
        str: Retrieved documents combined
    """
    # Create embedding for the query
    query_embedding = create_embeddings([query])[0]

    # Retrieve from vector database
    results = retrieve_from_vector_db(query_embedding, collection, n_results=n_results)

    # Combine retrieved documents
    if results['documents'] and len(results['documents'][0]) > 0:
        retrieved_docs = "\n\n".join(results['documents'][0])
        return retrieved_docs
    else:
        return "No relevant documents found."


def augment(retrieved_docs, query):
    """
    Augment the user query with retrieved context.

    Args:
        retrieved_docs: Retrieved documents from vector database
        query: User's question

    Returns:
        str: Augmented prompt
    """
    user_prompt = f"<context> {retrieved_docs} </context>\n\nBased on the above context, answer the user query: {query}"
    return user_prompt


def generate(user_prompt):
    """
    Generate response using LLM.

    Args:
        user_prompt: Augmented prompt with context

    Returns:
        str: Generated answer
    """
    messages = [
        {"role": "system", "content": "You are a helpful RAG assistant. Answer questions based on the provided context."},
        {"role": "user", "content": user_prompt}
    ]

    response = completion(
        model="openai/@openrouter-09cb28/openai/gpt-4.1-mini",
        messages=messages,
        api_key=os.getenv("PORTKEY_API_KEY"),
        api_base=os.getenv("PORTKEY_API_BASE"),
    )

    return response.choices[0].message['content']


if __name__ == "__main__":
    # Initialize ChromaDB client and collection
    client = get_chroma_client()
    collection = get_or_create_collection(client, "documents")

    # User queries to test
    queries = [
        "Who are the Dursleys?",
        "What is Harry Potter's story about?",
        "Where does Harry live?"
    ]

    for user_query in queries:
        print(f"\n{'='*70}")
        print(f"User Query: {user_query}")
        print("=" * 70)

        # Retrieval
        print("\n1. Retrieving relevant documents...")
        retrieved_docs = retrieve(user_query, collection, n_results=3)
        print(f"Retrieved Documents (first 500 chars):\n{retrieved_docs[:500]}...")

        # Augmentation
        print("\n2. Augmenting prompt...")
        user_prompt = augment(retrieved_docs, user_query)

        # Generation
        print("\n3. Generating answer...")
        answer = generate(user_prompt)
        print(f"\nGenerated Answer:\n{answer}")
        print(f"\n{'='*70}")
