import os
import PyPDF2
import unicodedata
from fuzzywuzzy import process

def normalize_text(text):
    """Normalize text to fix encoding issues (e.g., ligatures, special characters)."""
    return unicodedata.normalize("NFKC", text)

def extract_drug_name(text, drug_list):
    """Use fuzzy matching to find drug names in extracted text."""
    text = text.lower()  # Convert text to lowercase
    for drug in drug_list:
        match = process.extractOne(drug.lower(), text.split())
        if match and match[1] > 85:  # Match confidence threshold (85%)
            return match[0]
    return "Unknown"

def pdf_text():
    result = ""
    pdf_folder = os.path.join(os.path.dirname(__file__), "pdfs")

    if not os.path.exists(pdf_folder):
        print(f"The folder '{pdf_folder}' does not exist.")
        return ""

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        return "No PDF files found in the folder."

    extracted_text = ""
    
    # Process only one PDF file
    pdf_path = os.path.join(pdf_folder, pdf_files[0])
    print(f"Extracting text from: {pdf_files[0]}")

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            for page in reader.pages:
                extracted_text += page.extract_text() or ""  # Ensure text is not None

        extracted_text = normalize_text(extracted_text)  # Fix encoding issues
        return extracted_text

    except Exception as e:
        print(f"Error reading {pdf_files[0]}: {e}")
        return ""

if __name__ == "__main__":
    text = pdf_text()
    if text:
        print("\nExtracted Text:\n", text)

        # List of drug names to check
        drug_list = ["tadalafil"]
        detected_drug = extract_drug_name(text, drug_list)
        print("\nDetected Drug Name:", detected_drug)
