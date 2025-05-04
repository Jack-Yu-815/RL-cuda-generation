import torch
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from judge import CudaCodeJudge
from utils import construct_dataset


# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id  # to supress the Hugging Face warning during generation


# setup LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


# TODO: use the customized judge
judge = CudaCodeJudge(batch_eval=False)
# judge = PairRMJudge()

# load your own dataset. follow dataset format: https://huggingface.co/docs/trl/main/en/online_dpo_trainer
dataset = construct_dataset(test_split=0.35, seed=42)
train_dataset = dataset["train"]
# remove the 3rd datapoint from the dataset due to weird CUDA memory access error
train_dataset = train_dataset.select([i for i in range(len(dataset["train"])) if i != 2])

# train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:50]")
print(f"Loaded {len(train_dataset)} samples from the dataset.")


training_args = OnlineDPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    output_dir="CUDA-gen-OnlineDPO", 
    logging_steps=10,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    save_steps=10,
    max_length=4096,
    max_new_tokens=2048,
    push_to_hub=True,
)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, 
)
trainer.train()
