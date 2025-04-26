import torch
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import BasePairwiseJudge
from judge import CudaCodeJudge

# quantization config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
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
# judge = CudaCodeJudge()
judge = PairRMJudge()

# TODO: load your own dataset. follow dataset format: https://huggingface.co/docs/trl/main/en/online_dpo_trainer
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:50]")
print(f"Loaded {len(train_dataset)} samples from the dataset.")


training_args = OnlineDPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    output_dir="OnlineDPO", 
    logging_steps=10,
)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()