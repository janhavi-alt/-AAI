# Install required libraries if not already installed
# pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Use a custom cache directory to avoid interruptions
os.environ["HF_HOME"] = "./hf_cache"

# Load DialoGPT model (more suited for chatbots)
model_name = "microsoft/DialoGPT-medium"
try:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to chat with the bot
def chat_with_bot(input_text):
    try:
        # Encode the input text and add EOS token
        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

        # Generate a response from the model with improved settings
        outputs = model.generate(
            inputs,
            max_length=100,  # Limit the response length
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            temperature=0.7,  # Enable diverse sampling
            top_p=0.9,  # Nucleus sampling for better responses
            top_k=50,  # Top-k filtering for controlled randomness
            do_sample=True,  # Enable sampling for non-deterministic responses
        )

        # Decode the response back into text and filter it
        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return filter_response(response)
    except Exception as e:
        return f"Error during response generation: {e}"

# Function to filter out nonsensical responses
def filter_response(response):
    # Check if the response is too short or off-topic
    if len(response.strip()) < 20:  # Adjust this threshold based on your needs
        return "I didn't understand that, could you rephrase?"
    return response

# Start a conversation loop
if __name__ == "__main__":
    print("Chatbot: Hello! Type 'exit' to end the conversation.")
    
    while True:
        # Take input from the user
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        # Get the bot's response and print it
        response = chat_with_bot(user_input)
        print(f"Chatbot: {response}")
