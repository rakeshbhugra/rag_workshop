from litellm import embedding
from dotenv import load_dotenv
import os

load_dotenv()

def create_embeddings(text):
    response = embedding(
        model="text-embedding-3-small",
        input=text,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    embeddings = [item['embedding'] for item in response['data']]
    return embeddings


# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "good morning from litellm",
        "this is another text to embed",
        "embeddings are useful for semantic search"
    ]

    # Create embeddings
    embeddings = create_embeddings(texts)

    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"\nFirst embedding (first 10 values): {embeddings[0][:10]}")