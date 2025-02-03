import pandas as pd
import ollama
import chromadb
from chromadb.utils import embedding_functions
import json
from ocr import pdf_text
# Load the SentenceTransformer embedding function
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="drug_data", embedding_function=embedder)

# Function to load and store CSV data in ChromaDB
def load_csv_to_chroma(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Posts"])  # Ensure no empty posts

    # Get all existing documents and delete them
    all_docs = collection.get()
    if "ids" in all_docs and all_docs["ids"]:
        collection.delete(ids=all_docs["ids"])

    # Insert each row as a document in ChromaDB
    for idx, row in df.iterrows():
        metadata = {
            "Drug Name": row["Drug Name"],
            "Adverse effects": row["Adverse effects(Yes/No)"],
            "Severity": row["Severity"],
            "Side/Harmful effects": row["Side/Harmful effects"]
        }
        collection.add(
            ids=[str(idx)],
            documents=[row["Posts"]],
            metadatas=[metadata]
        )

    print(f"✅ Loaded {len(df)} records into ChromaDB.")

# Function to retrieve similar posts from ChromaDB
def retrieve_similar_posts(query, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    if results["documents"]:
        return results["metadatas"]
    return []

# Function to generate a prompt for Ollama
# Function to generate a prompt for Ollama
def generate_prompt(similar_posts):
    prompt = "You are a medical text-processing AI. Extract structured information from user input based on similar examples:\n"
    
    for example in similar_posts:  # example is a dictionary inside a list
        if isinstance(example, list):  # Fix: Extract the first dictionary if inside a list
            example = example[0]  # ChromaDB returns results as nested lists

        prompt += f"""
        Example:
        Output JSON: {{
            "Drug Name": "{example.get('Drug Name', 'Unknown')}",
            "Adverse effects": "{example.get('Adverse effects', 'Unknown')}",
            "Severity": "{example.get('Severity', 'Unknown')}",
            "Side/Harmful effects": "{example.get('Side/Harmful effects', 'Unknown')}"
        }}\n"""

    prompt += "\nAlways return a JSON object with the same structure based on the given input."
    return prompt


# Query Ollama with retrieved knowledge
def query_ollama(text, system_prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    print("*************************************************")
    return response['message']['content']  # Convert response to JSON

# Main Function
def main():
    csv_path = "dataset_shortened.csv"  # Change to your CSV file path
    load_csv_to_chroma(csv_path)  # Load CSV into ChromaDB

    while True:
        user_input = pdf_text()

        similar_posts = retrieve_similar_posts(user_input)
        if not similar_posts:
            print("⚠️ No relevant data found. Please try a different input.")
            continue

        system_prompt = generate_prompt(similar_posts)
        structured_output = query_ollama(user_input, system_prompt)

        print("\nExtracted Information (JSON):\n", structured_output)

if __name__ == "__main__":
    main()
