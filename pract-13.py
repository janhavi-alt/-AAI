# pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can change to a larger version like 'gpt2-medium' if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token to EOS token (GPT-2 doesn't use padding)
tokenizer.pad_token = tokenizer.eos_token

# Provide a prompt to generate text
prompt = "Once upon a time in a faraway land"
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(inputs, 
                         max_length=50,  # Adjust the length of generated text
                         num_return_sequences=5,  # Generate 5 different outputs
                         do_sample=True,  # Enable sampling for diversity
                         top_k=50,  # Top-k sampling
                         top_p=0.95,  # Nucleus sampling (top-p)
                         attention_mask=None,  # Attention mask (not required for GPT-2)
                         pad_token_id=tokenizer.eos_token_id)  # Set pad_token_id to eos_token_id

# Print generated texts
print("\nGenerated Texts:")
for i, output in enumerate(outputs, start=1):
    print(f"{i}: {tokenizer.decode(output, skip_special_tokens=True)}")
