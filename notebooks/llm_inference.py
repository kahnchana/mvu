from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM

from utils import get_ego_schema, calc_loglikelihood


if __name__ == "__main__":

    dataset = get_ego_schema()

    device = "cuda" # the device to load the model onto
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    save_dict = {}

    correct, total = 0, 0
    for datum in tqdm(dataset):
        only_prompts = tokenizer(datum['only_prompts'])
        prompt_lengths = [len(x) - 1 for x in only_prompts['input_ids']]
        encoded = tokenizer(datum['prompts'], return_tensors="pt", padding=True)
        labels = encoded['input_ids'].clone()
        for idx, length in enumerate(prompt_lengths):
            labels[idx, :-length] = -100
        model_inputs = {x: y.to(device) for x,y in encoded.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs)
        loss = calc_loglikelihood(model_outputs.logits.detach(), labels)
        pred = loss.argmin().item()
        correct += datum['ans'] == pred
        total += 1
        if (total + 1) % 100 == 0:
            print(f"Accuracy: {correct / total}")

    
    print(f"Final Accuracy: {100 * correct / total} %")