import pandas as pd
import ollama
import json
from ocr import pdf_text
# Load the CSV file and extract format mappings
def load_format(csv_path):
    df = pd.read_csv(csv_path)
    format_examples = df[["Posts", "Drug Name", "Adverse effects(Yes/No)", "Severity", "Side/Harmful effects"]].dropna()
    print("Data loaded")
    return format_examples.to_dict(orient="records")

# Generate a system prompt based on CSV examples
def generate_prompt(format_examples):
    prompt = "You are a medical text-processing AI. Extract structured information from user input based on these examples:\n"
    print("into system prompt")
    for example in format_examples:
        prompt += f"""
        Example:
        Post: {example["Posts"]}
        Output format: {{
            "Drug Name": "{example["Drug Name"]}",
            "Adverse effects": "{example["Adverse effects(Yes/No)"]}",
            "Severity": "{example["Severity"]}",
            "Side/Harmful effects": "{example["Side/Harmful effects"]}"
        }}\n"""
    prompt += "\ndont given any other text as output apart from the output format given above. Always return a JSON object with the same structure based on the output format given above. "
    return prompt

# Query Ollama with the formatted prompt
def query_ollama(text, system_prompt):
    response = ollama.chat(
        model="llama3.2",  # Change to your local Ollama model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    print(response['message']['content'])
    return  response['message']['content'] # Convert response to JSON

def main():
    csv_path = "dataset_shortened.csv"  # Path to your CSV file
    format_examples = load_format(csv_path)
    system_prompt = generate_prompt(format_examples)

    while True:
        # user_input = input("Enter a medical post (or type 'exit' to quit): ")
        # if user_input.lower() == "exit":
        #     break
        user_input = pdf_text()
        structured_output = query_ollama(user_input, system_prompt)
        print("\nExtracted Information (JSON):\n", json.dumps(structured_output, indent=4))

if __name__ == "__main__":
    main()
