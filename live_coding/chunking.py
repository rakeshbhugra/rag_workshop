def chunk_text(text, chunk_size=500):
    chunks = []
    start = 0
    
    while start < len(text.split()):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk.strip())

        start += chunk_size

    return chunks

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text to demonstrate chunking. " * 20

    chunks = chunk_text(sample_text, chunk_size=100)

    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk)} words):")
        print(chunk)