import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataset_1 import build_prompt,make_dataset
from eval import evaluate_tool_call




# Prefer the local model (downloaded by 0-download_model.py)
LOCAL_MODEL_DIR = "../dpo/Qwen2.5-0.5B-Instruct"
model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test prompt: the user states a biased opinion (this prompt is not in the training data)
examples = make_dataset()
predictions =[]
for example in examples:
    messages = [{"role": "user", "content": build_prompt(example["prompt"])}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Test the base output before alignment
    outputs = model.generate(**inputs, max_new_tokens=500)
    print("=" * 40)
    print("[Original response before fine-tuning]")
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(response)
    print("=" * 40)
    predictions.append(response)

eval_result = evaluate_tool_call(predictions=predictions,examples=examples)
print(eval_result)