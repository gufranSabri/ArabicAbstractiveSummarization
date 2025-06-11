import pandas as pd
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, AutoTokenizer
from sklearn.model_selection import train_test_split
from rouge import Rouge
import numpy as np
from tqdm import tqdm

# Load datasets
train_df = pd.read_csv('./data/train1_processed.csv').dropna()
val_df = pd.read_csv('./data/valid_processed.csv').dropna()
test_df = pd.read_csv('./data/test_processed.csv').dropna()
ood_df = pd.read_csv('./data/ood_processed.csv').dropna()

# Extract text and targets
train_texts = train_df['text_preprocessed'].astype(str).tolist()
train_targets = train_df['headline'].astype(str).tolist()
val_texts = val_df['text_preprocessed'].astype(str).tolist()
val_targets = val_df['headline'].astype(str).tolist()
test_texts = test_df['text_preprocessed'].astype(str).tolist()
test_targets = test_df['headline'].astype(str).tolist()
ood_texts = ood_df['text_preprocessed'].astype(str).tolist()
ood_targets = ood_df['headline'].astype(str).tolist()

# Load tokenizer and model
model_name = "UBC-NLP/AraT5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 40
BATCH_SIZE = 8

# Tokenization function
def tokenize_function(texts, targets):
    input_encodings = tokenizer(
        texts,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
    labels = target_encodings["input_ids"]
    # Replace pad token id's in labels with -100 so loss ignores padding
    labels = tf.where(labels == tokenizer.pad_token_id, -100, labels)
    return input_encodings, labels

# Tokenize datasets
train_inputs, train_labels = tokenize_function(train_texts, train_targets)
val_inputs, val_labels = tokenize_function(val_texts, val_targets)
test_inputs, test_labels = tokenize_function(test_texts, test_targets)
ood_inputs, ood_labels = tokenize_function(ood_texts, ood_targets)

# Create tf.data.Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": train_inputs["input_ids"],
        "attention_mask": train_inputs["attention_mask"],
    },
    train_labels
)).shuffle(1000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": val_inputs["input_ids"],
        "attention_mask": val_inputs["attention_mask"],
    },
    val_labels
)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": test_inputs["input_ids"],
        "attention_mask": test_inputs["attention_mask"],
    },
    test_labels
)).batch(BATCH_SIZE)

ood_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": ood_inputs["input_ids"],
        "attention_mask": ood_inputs["attention_mask"],
    },
    ood_labels
)).batch(BATCH_SIZE)

# Custom wrapper model with train and test step
class T5ModelWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self.model(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"],
                labels=y,
                training=True
            )
            loss = outputs.loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            labels=y,
            training=False
        )
        return {"loss": outputs.loss}

# Function to generate summaries and compute ROUGE-L F1 score
def compute_rouge_l(model, dataset, tokenizer, max_length=40):
    rouge = Rouge()
    rouge_l_f1_scores = []

    for batch in tqdm(dataset, desc="Computing ROUGE-L"):
        input_ids = batch[0]["input_ids"]
        attention_mask = batch[0]["attention_mask"]
        labels = batch[1]

        # Generate predictions
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode targets (labels)
        labels = labels.numpy()
        # Replace -100 with pad_token_id to decode properly
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE-L F1 for each prediction-target pair
        for pred, target in zip(preds, targets):
            scores = rouge.get_scores(pred, target)[0]
            rouge_l_f1_scores.append(scores['rouge-l']['f'])

    if len(rouge_l_f1_scores) == 0:
        return 0.0
    return np.mean(rouge_l_f1_scores)

# Logger function (you can replace with actual logger if needed)
def logger(msg):
    print(msg)

# Wrap and compile model
wrapped_model = T5ModelWrapper(model)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
wrapped_model.compile(optimizer=optimizer)

# Training with evaluation after each epoch
TOTAL_TRAIN_STEPS = 66000
EPOCHS = int(TOTAL_TRAIN_STEPS / len(train_dataset)) + 1

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    # wrapped_model.fit(train_dataset, validation_data=val_dataset, epochs=1)

    # Evaluate ROUGE-L on test and OOD datasets
    test_rouge_l = compute_rouge_l(wrapped_model, test_dataset, tokenizer)
    ood_rouge_l = compute_rouge_l(wrapped_model, ood_dataset, tokenizer)

    logger(f"Test ROUGE-L F1: {test_rouge_l:.4f}")
    logger(f"OOD ROUGE-L F1: {ood_rouge_l:.4f}")
