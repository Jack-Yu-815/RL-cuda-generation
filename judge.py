from trl import BasePairwiseJudge
import random
from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import extract_first_code, extract_last_code

# Define a custom judge. reference https://huggingface.co/docs/trl/main/en/judges#define-your-own-judge
class CudaCodeJudge(BasePairwiseJudge):
    def judge(self, prompts, completions, shuffle_order=False):
        better_indices = []
        for prompt in prompts:
            results = []
            for completion in completions:
                # 1. Check if the prompt is valid
                print(f"Prompt: {prompt}")
                print(f"Completion: {completion}")
                # if not isinstance(prompt, str) or not isinstance(completion, str):
                #     raise ValueError("Prompt and completion must be strings")
                custom_cuda = extract_first_code(completion, ["python", "cpp"])
                ref_arch_src = extract_last_code(prompt, ["python", "cpp"])
                
                # check LLM is able to generate custom CUDA code
                assert custom_cuda is not None, "Custom CUDA code generation failed"
                
                kernel_exec_result = eval_kernel_against_ref(
                    ref_arch_src, custom_cuda, verbose=False, measure_performance=True, num_correct_trials=5, num_perf_trials=100
                )

                results.append(kernel_exec_result)
            
            result1, result2 = results
            # 2. TODO: FIXME!! Compare the results
            better_indices.append(0)
        
        return better_indices