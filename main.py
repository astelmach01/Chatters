from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

def get_device() -> str:
    if torch.cuda.is_available():
        logging.info(f"Running CUDA on {torch.cuda.device_count()} devices")
        return "cuda"
    elif torch.backends.mps.is_available():
        logging.info("Running on MPS")
        return "mps"
    else:
        logging.info("Running on CPU")
        return "cpu"
    
device = get_device()

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


def chat(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]