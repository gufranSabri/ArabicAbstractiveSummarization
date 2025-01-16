import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define model name and load the model and tokenizer
model_name = "UBC-NLP/AraT5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)

# Define a dummy input
dummy_input = "اشرح معنى مرحبا كيف حالك؟"

# Tokenize the input
inputs = tokenizer(dummy_input, return_tensors="pt", max_length=512, truncation=True)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=50, 
        num_beams=2, 
        early_stopping=True
    )

print(outputs)

# Decode the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the results
print("Input:", dummy_input)
print("Generated Output:", decoded_output)

# Assertions for testing
assert isinstance(decoded_output, str), "Output should be a string."
assert len(decoded_output) > 0, "Output should not be empty."

print("Dummy test passed!")