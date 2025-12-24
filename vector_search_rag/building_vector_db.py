'''
input - pdf
parse the pdf to extract text
chunking
create embeddings - openai
build vector db

'''

from parse_pdf_pages import parse_pdf_pages
from chunking import chunk_text
from embeddings import create_embeddings
from chromadb_utils import (
    get_chroma_client,
    get_or_create_collection,
    add_chunk_to_chromadb
)


def add_document_to_vector_db(pdf_path, document_name, max_pages=10,
                               chunk_size=500, overlap=50,
                               collection_name="documents"):
    """
    Process a PDF document and add it to ChromaDB vector database.

    Args:
        pdf_path: Path to the PDF file
        document_name: Name to identify the document
        max_pages: Maximum number of pages to parse (default: 10)
        chunk_size: Size of each chunk in characters (default: 500)
        overlap: Number of overlapping characters between chunks (default: 50)
        collection_name: Name of the ChromaDB collection (default: "documents")

    Returns:
        int: Number of chunks added to the database
    """
    print(f"Processing document: {document_name}")
    print("=" * 70)

    # Step 1: Parse PDF
    print("\n1. Parsing PDF...")
    pages = parse_pdf_pages(pdf_path, max_pages=max_pages)

    # Combine all page texts
    full_text = "\n\n".join([page['text'] for page in pages if page['text'].strip()])
    print(f"   Total characters extracted: {len(full_text)}")

    # Step 2: Chunk text
    print("\n2. Chunking text...")
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Create embeddings
    print("\n3. Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"   Created {len(embeddings)} embeddings")

    # Step 4: Initialize ChromaDB
    print("\n4. Adding to ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)

    # Step 5: Add chunks to ChromaDB
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        add_chunk_to_chromadb(chunk, embedding, document_name, idx, collection)

    print(f"\n✓ Successfully added {len(chunks)} chunks to vector database")
    print(f"✓ Total documents in collection: {collection.count()}")

    return len(chunks)


# Example usage
if __name__ == "__main__":
    pdf_path = "../input_data/harrypotter.pdf"
    document_name = "harry_potter"

    # Process and add document to vector DB
    num_chunks = add_document_to_vector_db(
        pdf_path=pdf_path,
        document_name=document_name,
        max_pages=20,
        chunk_size=500,
        overlap=50
    )

    print(f"\n{'='*70}")
    print(f"Processing complete! Added {num_chunks} chunks to the database.")
