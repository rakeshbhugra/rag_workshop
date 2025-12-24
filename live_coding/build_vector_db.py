'''
RAG

Building the Vector Database
PDF -> Text Parsing (PyMuPDF) -> Chunking -> embeddings (OpenAI) -> Vector DB (ChomaDB) 

Retrieval 
query -> embedding -> vector search -> relevant docs

Augmented

Generation

'''


def pdf_to_text(pdf_path, max_pages=20):
    pass


def chunking(text):
    pass


def create_embeddings(chunks):
    pass


def add_document_to_vector_db(pdf_path):
    text = pdf_to_text(pdf_path)

    chunks = chunking(text)

    embeddings = create_embeddings(chunks)

    # Add to vector DB
    pass