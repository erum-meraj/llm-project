import pandas as pd
import ollama
import json
import textwrap
import re
from ocr import pdf_text

# Load CSV format examples
def load_format(csv_path):
    df = pd.read_csv(csv_path)
    return df.dropna().to_dict(orient="records")

# Generate system prompt ensuring structured JSON output
def generate_prompt(format_examples):
    prompt = (
        "You are a medical text-processing AI. Extract structured information from the provided input and "
        "format it as a JSON object with the following fields:\n\n"
        "1. Links: A URL or reference link (if available).\n"
        "2. Posts: The original text input.\n"
        "3. Preprocessed Posts: A cleaned and shortened version of the text.\n"
        "4. Drug Name: The name of the drug mentioned in the text.\n"
        "5. Adverse effects(Yes/No): Whether adverse effects are mentioned (Yes/No).\n"
        "6. Severity: The severity of the adverse effects (Mild/Moderate/Severe).\n"
        "7. Side/Harmful effects: A description of the side or harmful effects.\n"
        "8. Images(Physical/Non physical): Whether images are physical or non-physical.\n\n"
        "Always return valid JSON strictly matching this format. Do not include any additional fields, explanations, or text outside the JSON object.\n\n"
        "Example Output:\n"
        "{\n"
        '  "Links": "N/A",\n'
        '  "Posts": "Original text input...",\n'
        '  "Preprocessed Posts": "Cleaned and shortened text...",\n'
        '  "Drug Name": "Tadalafil",\n'
        '  "Adverse effects(Yes/No)": "Yes",\n'
        '  "Severity": "Moderate",\n'
        '  "Side/Harmful effects": "Bullous fixed drug eruption",\n'
        '  "Images(Physical/Non physical)": "Physical"\n'
        "}\n\n"
    )

    for example in format_examples:
        prompt += (
            f"\nExample Input:\n{example['Posts']}\n"
            f"Example Output:\n{json.dumps(example, indent=4)}\n"
        )

    return prompt

# Preprocess and clean text for model input
def preprocess_text(text, max_length=2000):
    text = re.sub(r'\s+', ' ', text).strip()
    return textwrap.shorten(text, width=max_length, placeholder="...")

# Query Ollama with streaming response handling
def query_ollama(text, system_prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            response_text += chunk['message']['content']

    # Debug: Print the raw response from the model
    print("Raw Model Response:\n", response_text)

    # Attempt to extract JSON from the response
    try:
        # Look for the first occurrence of '{' and the last occurrence of '}'
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("No JSON object found in the response.")

        # Extract the JSON portion
        json_str = response_text[json_start:json_end]
        output = json.loads(json_str)

        # Validate required fields
        required_fields = [
            "Links", "Posts", "Preprocessed Posts", "Drug Name",
            "Adverse effects(Yes/No)", "Severity", "Side/Harmful effects",
            "Images(Physical/Non physical)"
        ]
        for field in required_fields:
            if field not in output:
                output[field] = "N/A"
        return output
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON Parsing Error: {e}")
        # Fallback to default structure if JSON is invalid
        return {
            "Links": "N/A",
            "Posts": text,
            "Preprocessed Posts": preprocess_text(text),
            "Drug Name": "Unknown",
            "Adverse effects(Yes/No)": "No",
            "Severity": "Mild",
            "Side/Harmful effects": "None",
            "Images(Physical/Non physical)": "Non physical"
        }

def main():
    csv_path = "dataset_shortened.csv"
    format_examples = load_format(csv_path)
    system_prompt = generate_prompt(format_examples)

    # Extract text from PDF
    user_input = pdf_text()
    if not user_input:
        print("No text extracted from PDF.")
        return

    # Preprocess the extracted text
    user_input = preprocess_text(user_input)

    # Query Ollama for structured output
    structured_output = query_ollama(user_input, system_prompt)

    # Convert the structured output to a DataFrame
    output_df = pd.DataFrame([structured_output])

    # Save the DataFrame to a CSV file
    output_df.to_csv("output.csv", index=False)

    # Print the final JSON output
    print(json.dumps(structured_output, indent=4))

if __name__ == "__main__":
    main()