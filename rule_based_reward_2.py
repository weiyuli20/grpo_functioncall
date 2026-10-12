
import json
from typing import Optional
from dataset_1 import execute_tool


def parse_call(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def schema_ok(call: dict, gold: dict) -> bool:
    if call.get("name") != gold.get("name"):
        return False
    if not isinstance(call.get("arguments"), dict):
        return False
    for key, gold_value in gold["arguments"].items():
        if key not in call["arguments"]:
            return False
        if not isinstance(call["arguments"][key], type(gold_value)):
            return False
    return True


def tool_reward(completions, gold_call, expected_result, **kwargs):
    rewards = []
    for completion, gold_raw, expected in zip(completions, gold_call, expected_result):
        gold = json.loads(gold_raw)
        call = parse_call(completion)
        if call is None:
            rewards.append(0.0)
            continue

        reward = 0.2  #大体格式正确
        if call.get("name") == gold["name"]:
            reward += 0.3  #工具调用正确
        if schema_ok(call, gold):
            reward += 0.3 #参数格式和参数值

        if gold["name"] == "no_call":
            if call.get("name") == "no_call":
                reward += 0.2  #相当于有工具时的结果正确
            rewards.append(reward)
            continue

        try:
            result = execute_tool(call["name"], call["arguments"])
            if str(result) == expected:
                reward += 0.2
        except Exception:
            pass

        rewards.append(reward)
    return rewards