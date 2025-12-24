import chromadb
from chromadb.config import Settings


# Initialize ChromaDB client
def get_chroma_client(persist_directory="./chroma_db"):
    """
    Initialize and return ChromaDB client.

    Args:
        persist_directory: Directory to persist the database

    Returns:
        chromadb.Client: ChromaDB client instance
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def get_or_create_collection(client, collection_name="documents"):
    """
    Get or create a ChromaDB collection.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection

    Returns:
        Collection: ChromaDB collection
    """
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def add_chunk_to_chromadb(chunk, embedding, document_name, idx, collection):
    """
    Add a single chunk with its embedding to ChromaDB.

    Args:
        chunk: Text chunk to store
        embedding: Embedding vector for the chunk
        document_name: Name of the source document
        idx: Index of the chunk
        collection: ChromaDB collection to add to
    """
    # Create unique ID for this chunk
    chunk_id = f"{document_name}_chunk_{idx}"

    # Add to collection
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{
            "document_name": document_name,
            "chunk_index": idx
        }]
    )

    print(f"Added chunk {idx} from {document_name} to ChromaDB")


def retrieve_from_vector_db(query_embedding, collection, n_results=5):
    """
    Retrieve the most relevant chunks from ChromaDB based on query embedding.

    Args:
        query_embedding: Embedding vector for the query
        collection: ChromaDB collection to search in
        n_results: Number of results to return (default: 5)

    Returns:
        dict: Results containing documents, distances, and metadatas
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


# Example usage
if __name__ == "__main__":
    # Initialize client and collection
    client = get_chroma_client()
    collection = get_or_create_collection(client, "test_collection")

    # Sample data
    sample_chunk = "This is a sample text chunk for testing."
    sample_embedding = [0.1] * 1536  # Mock embedding (text-embedding-3-small has 1536 dimensions)

    # Add to ChromaDB
    add_chunk_to_chromadb(
        chunk=sample_chunk,
        embedding=sample_embedding,
        document_name="test_document.pdf",
        idx=0,
        collection=collection
    )

    print(f"\nTotal documents in collection: {collection.count()}")
