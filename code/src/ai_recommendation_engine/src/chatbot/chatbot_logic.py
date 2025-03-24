from transformers import pipeline

def generate_chatbot_response(prompt, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    pipe = pipeline("text-generation", model=model_name)
    response = pipe(prompt, max_new_tokens=500)[0]['generated_text']
    return response
