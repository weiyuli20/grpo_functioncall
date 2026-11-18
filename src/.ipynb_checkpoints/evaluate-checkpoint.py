import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设这些是你重构后的模块路径
from rule_based_reward import parse_call, schema_ok, execute_tool
from dataset import build_prompt, make_dataset

def evaluate_tool_call(predictions: list[str], examples: list[dict]) -> dict:
    """计算各项评测指标"""
    total = len(examples)
    valid_json = 0
    name_match = 0
    schema_match = 0
    exact_match = 0

    for pred, ex in zip(predictions, examples):
        try:
            gold = json.loads(ex["gold_call"])
        except json.JSONDecodeError:
            print(f"Warning: Gold standard for prompt is not valid JSON: {ex['gold_call']}")
            continue

        call = parse_call(pred)
        if call is None:
            continue

        valid_json += 1
        if call.get("name") == gold.get("name"):
            name_match += 1
        if schema_ok(call, gold):
            schema_match += 1

        try:
            if gold["name"] == "no_call":
                if call.get("name") == "no_call":
                    exact_match += 1
            else:
                result = execute_tool(call["name"], call["arguments"])
                if str(result) == str(ex["expected_result"]):
                    exact_match += 1
        except Exception:
            pass

    # 防止除零异常
    if total == 0:
        return {}

    return {
        "response_validity": round(valid_json / total, 4),
        "function_name_accuracy": round(name_match / total, 4),
        "schema_match": round(schema_match / total, 4),
        "exact_match": round(exact_match / total, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on financial tool-calling tasks.")
    parser.add_argument("--model_path", type=str, default="../Qwen3-1.7B", 
                        help="Path to the fine-tuned model or HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Where to save the results")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 加上 padding_side="left" 方便 batch generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()

    print("Preparing dataset...")
    examples = make_dataset()
    
    # 将所有 prompt 预处理为 chat 格式
    texts = []
    for example in examples:
        messages = [{"role": "user", "content": build_prompt(example["prompt"])}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking = False)
        texts.append(text)

    print(f"Generating predictions for {len(texts)} examples...")
    predictions = []
    
    # 使用 Batch 进行生成，大幅加速
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Evaluating"):
        batch_texts = texts[i:i + args.batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # 评测时使用贪心解码，保证结果可复现
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                
            )

        # 截取生成的内容（去掉输入的 prompt 部分）
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        # 按 batch 解码
        batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 打印第一个样本观察一下
        if i == 0:
            print("\n" + "=" * 40)
            print("[Sample Prediction]")
            print(batch_predictions[0])
            print("=" * 40 + "\n")
            
        predictions.extend(batch_predictions)

    print("Calculating metrics...")
    eval_result = evaluate_tool_call(predictions=predictions, examples=examples)
    
    # 打印并保存结果
    print(json.dumps(eval_result, indent=4))
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_path, "metrics": eval_result}, f, indent=4)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()