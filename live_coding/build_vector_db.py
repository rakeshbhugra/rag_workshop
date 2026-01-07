'''
RAG

Building the Vector Database
PDF -> Text Parsing (PyMuPDF) -> Chunking -> embeddings (OpenAI) -> Vector DB (ChomaDB) 

Retrieval 
query -> embedding -> vector search -> relevant docs

Augmented

Generation

'''
from parse_pdf_pages import parse_pdf_pages
from chunking import chunk_text
from create_embeddings import create_embeddings
import chromadb


client = chromadb.Client()

def add_chunk_to_chroma_db(chunk, embedding, chunk_index, source):
    chunk_id = f"{source}_chunk_{chunk_index}"
    collection = client.get_or_create_collection(name="documents")

    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": source, "chunk_index": chunk_index}]
    )

    return chunk_id

def add_document_to_vector_db(pdf_path):
    pages_data = parse_pdf_pages(pdf_path)
    full_text = "\n\n".join([page['text'] for page in pages_data if page['text'].strip()])

    chunks = chunk_text(full_text)

    embeddings = create_embeddings(chunks)

    # Extract document name from path for unique IDs
    doc_name = pdf_path.split('/')[-1].replace('.pdf', '')

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        add_chunk_to_chroma_db(chunk, embedding, idx, doc_name)

    print("done...")

    return len(chunks)


def retrive_from_vector_db(query):
    collection = client.get_or_create_collection(name="documents")
    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    return results


if __name__ == "__main__":
    pdf_path = "/Users/rakeshbhugra/code/qure/workshop/rag_workshop/input_data/harrypotter.pdf"
    num_chunks = add_document_to_vector_db(pdf_path)
    print(f"Total chunks added to vector DB: {num_chunks}")

    query = "What is the main theme of the book?"
    results = retrive_from_vector_db(query)
    print("Retrieval results:")
    # ChromaDB query returns a dict with 'documents', 'ids', 'metadatas', etc.
    # results['documents'][0] contains the list of retrieved documents
    for doc in results['documents'][0]:
        print(f" - {doc}")