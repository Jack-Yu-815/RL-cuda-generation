import torch
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
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
# lora_dir = "/data/user_data/amittur/dapt_5_epochs/epoch_3"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
model = prepare_model_for_kbit_training(model)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
# )
# print(model)
# model = PeftModel.from_pretrained(model, lora_dir)
# print(model)
# model = peft_model.merge_and_unload()

# print(model)

# print("Loading base model for reference (device_map='cpu')...")
# ref_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # device_map="cpu", # Force reference model to CPU memory
#     device_map="auto",
#     torch_dtype=torch.bfloat16 # Keep dtype consistent if possible, but it will run on CPU
# )
# print("Reference model loaded (on CPU).")

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


# Instantiate the custom judge
judge = CudaCodeJudge()


# load your own dataset. follow dataset format: https://huggingface.co/docs/trl/main/en/online_dpo_trainer
dataset = construct_dataset(test_split=0.35, seed=42)
# TODO: use the customized judge
judge = CudaCodeJudge(batch_eval=False)
# judge = PairRMJudge()

# load your own dataset. follow dataset format: https://huggingface.co/docs/trl/main/en/online_dpo_trainer
dataset = construct_dataset(test_split=0.35, seed=42)
train_dataset = dataset["train"]
# remove the 3rd datapoint from the dataset due to weird CUDA memory access error
train_dataset = train_dataset.select([i for i in range(len(dataset["train"])) if i != 2])

# train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:50]")
print(f"Loaded {len(dataset)} samples from the dataset.")


# training_args = OnlineDPOConfig(
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     output_dir="OnlineDPO", 
#     logging_steps=1,
#     # max_new_tokens=5000, 
#     # missing_eos_penalty=1.0,
# )
# print(f"Training arguments: {training_args}")

training_args = OnlineDPOConfig(
    num_train_epochs=1,
    #per_device_train_batch_size=2, # Keep batch size at 1
    #gradient_accumulation_steps=4, # Use gradient accumulation
    #gradient_checkpointing=True,   # <<< Enable Gradient Checkpointing
    #output_dir="OnlineDPO_no_quant_cpu_ref",
    #logging_steps=10, # Log less frequently than every step if bottlenecked
    #learning_rate=5e-6,
    #optim="adamw_torch_fused",
    #report_to="wandb", # Optional
    #remove_unused_columns=False,
    #bf16=True, # Use mixed precision for the active model on GPU
    # fp16=True, # Use fp16 if bf16 not supported
    # max_new_tokens=5000,        # Adjust as needed
    # max_new_tokens=1024,      # <<< REDUCED: Max tokens to generate in each step
                              # Make sure this is reasonable for CUDA kernels.
    # max_length=4096,        # <<< ADDED: Max total sequence length (prompt + completion)
                              # Ensure this fits in GPU memory and model context window.
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
print(f"Training arguments: {training_args}")


trainer = OnlineDPOTrainer(
    model=model, 
    judge=judge, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=dataset['train'],
    # eval_dataset=dataset['test'],
)
print(f"Trainer initialized with model: {model} and judge: {judge}")

print("Initializing OnlineDPOTrainer...")
# trainer = OnlineDPOTrainer(
#     model=model,            # Active model (GPU/auto)
#     ref_model=ref_model,    
#     judge=judge,
#     args=training_args,
#     processing_class=tokenizer,
#     train_dataset=dataset['train'].select(range(5)),
#     # eval_dataset=dataset['test'],
# )
# print(f"Trainer initialized.")

    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, 
)
trainer.train()
