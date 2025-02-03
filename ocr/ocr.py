import os
import PyPDF2

def pdf_text():
    result = ""
    # Define the path to the "pdfs" folder
    pdf_folder = os.path.join(os.path.dirname(__file__), "pdfs")

    # Check if the folder exists
    if not os.path.exists(pdf_folder):
        print(f"The folder '{pdf_folder}' does not exist.")
        exit()

    # Get a list of all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    # Check if there are any PDF files in the folder
    if not pdf_files:
        return ("No PDF files found in the folder.")
        exit()
    cnt = 0
    # Loop through each PDF file and extract text
    for pdf_file in pdf_files:
        if cnt == 1:
            break
        cnt+=1
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Extracting text from: {pdf_file}")
        

        try:
            # Open the PDF file
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                # Extract text from each page
                for page in reader.pages:
                    text += page.extract_text()

                # Print the extracted text
                # print(f"Text from {pdf_file}:\n{text}\n{'-'*50}\n")
                return text
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}\n")
            return ""
        

pdf_text()