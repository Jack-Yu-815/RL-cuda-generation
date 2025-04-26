import datasets
# from src.eval import KernelExecResult
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

def construct_dataset(test_split, seed=42):
    bench_dataset = datasets.load_dataset("ScalingIntelligence/KernelBench")
    split_datasets = {"train": [], "test": []}

    # for each level split, do a 60/40 train/test split
    for split_name, ds in bench_dataset.items():
        # create input prompt
        ds = ds.map(lambda x: {
            "prompt": [{"role": "user", "content": prompt_generate_custom_cuda_from_prompt_template(x['code'])}]
        })
        parts = ds.train_test_split(test_size=test_split, seed=seed)
        split_datasets["train"].append(parts["train"])
        split_datasets["test"].append(parts["test"])

    # concatenate all levels into one train & one eval dataset
    train_set = datasets.concatenate_datasets(split_datasets["train"])
    eval_set  = datasets.concatenate_datasets(split_datasets["test"])
    total_set = datasets.DatasetDict({"train": train_set, "test": eval_set})

    print(f"Training set size:   {len(total_set['train'])}")
    print(f"Test set size: {len(total_set['test'])}")

    return total_set