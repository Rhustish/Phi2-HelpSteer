from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Function to generate a response
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id, top_k=50, do_sample=True, repetition_penalty=1.3, top_p=0.95)
    response = tokenizer.decode(output[0])
    return response

# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt)
print(response)
