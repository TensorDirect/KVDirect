import torch
import requests
from transformers import AutoTokenizer
import argparse
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    with open(args.text_file, "r") as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt")
    assert len(inputs["input_ids"][0]) >= args.prompt_length, "Prompt is short" + str(len(inputs["input_ids"][0]))
    input_ids = inputs["input_ids"][0]
    
    prompt_id = input_ids[:args.prompt_length]
    print(len(input_ids))
    
    prompt = tokenizer.decode(prompt_id)
    
    res = requests.post("http://localhost:8000/generate", json={"prompt": prompt, "temperature": 0.0, "max_tokens": args.response_length})
    text = res.json()["text"][0]
    print(text[len(prompt):])
    text_id = tokenizer(text, return_tensors="pt")["input_ids"][0][args.prompt_length:]
    print(len(text_id))
    

    


