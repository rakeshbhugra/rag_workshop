import pymupdf

def parse_pdf_pages(pdf_path, max_pages=50):
    doc = pymupdf.open(pdf_path)

    total_pages = len(doc)
    print(f"Total pages in PDF: {total_pages}")


    pages_to_parse = min(max_pages, total_pages)
    print(f"Parsing first {pages_to_parse} pages...\n")

    pages_data = []
    for page_num in range(pages_to_parse):
        page = doc[page_num]

        text = page.get_text("text")

        print(text)

        pages_data.append({
            "page_number": page_num + 1,
            "text": text,
        })
        
    print(len(pages_data))
    doc.close()
    return pages_data

    
if __name__ == "__main__":
    pdf_path = "../input_data/harrypotter.pdf"
    parse_pdf_pages(pdf_path)