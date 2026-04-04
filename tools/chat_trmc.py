import torch
from unsloth import FastLanguageModel

def trmc_chat():
    model_path = "outputs/trmc_final_run"
    print(f"Loading TRMC Core from {model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 32768,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    while True:
        user_input = input("Architect Daniel > ")
        if user_input.lower() in ["exit", "quit"]: break
        
        inputs = tokenizer([user_input], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        response = tokenizer.batch_decode(outputs)
        
        print(f"\nTRMC MoE > {response[0]}")
        print("-" * 30)

if __name__ == "__main__":
    trmc_chat()