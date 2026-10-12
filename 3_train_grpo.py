from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from dataset_1 import make_dataset
from rule_based_reward_2 import tool_reward
from transformers import  AutoTokenizer, AutoModelForCausalLM
import os


LOCAL_MODEL_DIR = "../dpo/Qwen2.5-0.5B-Instruct"



model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading base model {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = make_dataset()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

paper_grpo_options = dict(
    beta=0.04,
    epsilon=0.2,
    num_iterations=1,
    scale_rewards="group",
    importance_sampling_level="token",
    loss_type="grpo",
)
if "use_bias_correction_kl" in GRPOConfig.__dataclass_fields__:
    paper_grpo_options["use_bias_correction_kl"] = False

training_args = GRPOConfig(
    output_dir="outputs/financial-tool-grpo",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=2048,
    max_completion_length=256,
    temperature=0.8,
    **paper_grpo_options,
    logging_steps=1,
    save_steps=50,
    max_steps=100,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=tool_reward,
    peft_config=peft_config,
)

trainer.train()