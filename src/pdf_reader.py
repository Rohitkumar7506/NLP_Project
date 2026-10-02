
import pymupdf

def extractText(pdfPath):
    document = pymupdf.open(pdfPath)
    pages = []

    for pageNumber, page in enumerate(document):
        text = page.get_text()
        pages.append({
            "page": pageNumber + 1,
            "text": text
        })

    document.close()
    return pages
import pymupdf

def extractText(pdfPath):
    document = pymupdf.open(pdfPath)
    pages = []

    for pageNumber, page in enumerate(document):
        text = page.get_text()
        pages.append({
            "page": pageNumber + 1,
            "text": text
        })

    document.close()
    return pages