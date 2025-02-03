# Install dependencies
#!pip install transformers torch datasets evaluate rouge-score sacrebleu -q

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate
from tqdm import tqdm

# Load BLEU and ROUGE scorers
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Change filename if needed

# Select required columns
df = df[['Preprocessed Posts', 'Drug Name', 'Side/Harmful effects']]

# Load Flan-T5 Model
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Function to extract ADR
def extract_adr(post):
    prompt = f"Extract drug name, adverse effects, adversity, and severity from the following medical post: {post}"
    output = pipe(prompt, max_length=100, truncation=True)[0]['generated_text']
    return output

# Run inference
tqdm.pandas()
df['Generated Output'] = df['Preprocessed Posts'].progress_apply(extract_adr)

# Evaluate using BLEU & ROUGE
references = df['Side/Harmful effects'].astype(str).tolist()
predictions = df['Generated Output'].astype(str).tolist()

bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_score = rouge.compute(predictions=predictions, references=references)

print(f"Flan-T5 BLEU Score: {bleu_score['score']}")
print(f"Flan-T5 ROUGE Scores: {rouge_score}")

# Save results
df.to_csv("flan_t5_results.csv", index=False)
print("Results saved to flan_t5_results.csv")
