from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from arabert.preprocess import ArabertPreprocessor
import pandas as pd
from rouge import Rouge
from tqdm import tqdm

# Load Model & Tokenizer
model_name = "malmarjeh/t5-arabic-text-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Preprocessor
preprocessor = ArabertPreprocessor(model_name="")

# Load and clean data
test_data = pd.read_csv('./data/test_processed.csv').dropna()
ood_data = pd.read_csv('./data/ood_processed.csv').dropna()

# Function to evaluate dataset using ROUGE-1 and ROUGE-L
def evaluate_rouge(dataset, text_column='text', summary_column='headline'):
    rouge = Rouge()
    generated_summaries = []
    reference_summaries = []

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluating"):
        text = preprocessor.preprocess(row[text_column])
        reference = row[summary_column]

        try:
            output = summarizer(
                text,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=3,
                repetition_penalty=3.0,
                max_length=200,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )[0]['generated_text']
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            output = ""

        generated_summaries.append(output)
        reference_summaries.append(reference)

    # Calculate ROUGE scores
    rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"]
    }

# Evaluate test and OOD data
print("Evaluating on TEST set:")
test_scores = evaluate_rouge(test_data)
print(f"Test ROUGE-1: {test_scores['rouge-1']:.4f}")
print(f"Test ROUGE-L: {test_scores['rouge-l']:.4f}")

print("\nEvaluating on OOD set:")
ood_scores = evaluate_rouge(ood_data)
print(f"OOD ROUGE-1: {ood_scores['rouge-1']:.4f}")
print(f"OOD ROUGE-L: {ood_scores['rouge-l']:.4f}")
