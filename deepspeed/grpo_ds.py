import os
import re
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl import GRPOTrainer, GRPOConfig

max_seq_length = 2048

#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
# Paths for cached assets on the scratch filesystem
SCRATCH_ROOT = "/lustre/orion/stf218/scratch/1eh"
TOKENIZER_CACHE_DIR = os.path.join(SCRATCH_ROOT, "nemotron-models-cache-7b")
MODEL_CACHE_DIR = os.path.join(SCRATCH_ROOT, "nemotron-models-cache-7b")
DATA_CACHE_DIR = os.path.join(SCRATCH_ROOT, "data_cache_trl")

if not os.path.isdir(MODEL_CACHE_DIR):
    raise FileNotFoundError(f"Expected model snapshot under {MODEL_CACHE_DIR}. Download it on a login node first.")
if not os.path.isdir(TOKENIZER_CACHE_DIR):
    raise FileNotFoundError(f"Expected tokenizer snapshot under {TOKENIZER_CACHE_DIR}. Download it on a login node first.")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Model and tokenizer names
base_model_name = "nvidia/OpenReasoning-Nemotron-32B"
new_model_name = "or-32b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True
)

#,local_files_only=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_DIR,
    dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Data set
data_name = "trl-lib/tldr"
DATASET_LOCAL_DIRNAME = data_name.replace("/", "___")
LOCAL_ARROW_BASENAME = "guanaco-llama2-1k-train.arrow"


def load_training_split() -> Dataset:
    """Prefer a locally cached Arrow file to avoid outbound HF requests."""
    dataset_root = Path(DATA_CACHE_DIR) / DATASET_LOCAL_DIRNAME
    if dataset_root.is_dir():
        try:
            arrow_path = next(dataset_root.rglob(LOCAL_ARROW_BASENAME))
            print(f"Loading training data from local cache: {arrow_path}")
            return Dataset.from_file(str(arrow_path))
        except StopIteration:
            pass

    try:
        print("DATA_CACHE_DIR", DATA_CACHE_DIR)
        print("Local Arrow cache not found; falling back to HF cached dataset.")
        return load_dataset("trl-lib/tldr", split="train", cache_dir="/lustre/orion/stf218/scratch/1eh/data_cache_trl") #("mlabonne/guanaco-llama2-1k", split="train", cache_dir="/lustre/orion/stf218/scratch/1eh/data_cache") #load_from_disk('data_cache/mlabonne___guanaco-llama2-1k/data')
        '''load_dataset(
            data_name,
            split="train",
            cache_dir='data_cache/mlabonne___guanaco-llama2-1k/data', #DATA_CACHE_DIR,
            local_files_only=True
        )'''
    except Exception as offline_err:
        msg = (
            f"Unable to load dataset '{data_name}' from local cache under {DATA_CACHE_DIR}. "
            "Download it on a login node (with network access) before launching training."
        )
        raise RuntimeError(msg) from offline_err


def format_reward(completions, **kwargs):
    """Reward completions that follow the desired format"""
    # Example: Check if the completion follows a think-then-answer format
    pattern = r"<think>(.*?)</think>\s*(.*?)"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # Check if there's substantial content in both sections
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(
                    0.5
                )  # Partial reward for correct format but limited content
        else:
            rewards.append(0.0)  # No reward for incorrect format

    print("format_reward", rewards)
    return rewards


training_data = load_training_split()
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])

# Training Params
train_params = GRPOConfig(
    max_completion_length = max_seq_length,
    output_dir="./results_modified", #checkpoint_dir,
    per_device_train_batch_size=2,
    #per_device_eval_batch_size=2,                 # <-- NEW
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    max_steps=8,
    #num_train_epochs=5.0,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    logging_steps=1,

    # ---- Validation & Checkpointing (NEW) ----
    #eval_strategy="steps",
    #eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=50,
    #load_best_model_at_end=True,
    #metric_for_best_model="eval_loss",
    #greater_is_better=False,
    #group_by_length=False,

    #gradient_checkpointing=True,
    #gradient_checkpointing_kwargs={'use_reentrant': False},
    #run_name="openreasoning-ft-mn",
    #ddp_find_unused_parameters=False,
    fp16=True,
    report_to="tensorboard",   # keep offline; set to ['wandb'] if you enable W&B
    deepspeed="ds_config.json"
)
print("GRPO config set")


#from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    use_rslora=False,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
print("LoRA config set")
#model = get_peft_model(base_model, peft_parameters)
#base_model.print_trainable_parameters()

print("configuring trainer")

# Trainer with LoRA configuration
fine_tuning = GRPOTrainer(
    model=base_model,
    train_dataset=training_data,
    #eval_dataset=val_dataset_hf,            # <-- NEW
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params,
    reward_funcs=[format_reward],
)

print("configured trainer")

# Training
fine_tuning.train()
base_model.save_pretrained("finetuned_grpo_test")
