from trl import BasePairwiseJudge
import random
from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import extract_first_code, extract_last_code

# Define a custom judge. reference https://huggingface.co/docs/trl/main/en/judges#define-your-own-judge
class CudaCodeJudge(BasePairwiseJudge):
    def _score_completion(self, result: KernelExecResult) -> float:
        """Assigns a score based on compilation, correctness, and runtime."""
        if not result.compiled:
            return -2.0  # Heavily penalize non-compiling code
        if not result.correctness:
            return -1.0  # Penalize incorrect code
        
        # Reward correct code, potentially favoring faster execution
        # Using a large constant minus runtime to reward lower runtimes.
        # Adjust the constant 1000.0 based on typical runtimes.
        # Consider adding a small epsilon to avoid division by zero if runtime can be 0.
        score = 1.0 
        if result.runtime is not None and result.runtime > 0:
             # Simple reward: higher score for lower runtime. Invert runtime.
             # Add a base score for correctness. Scale runtime reward.
             score += 1.0 / (result.runtime + 1e-6) # Add epsilon for stability
        elif result.runtime == 0:
             score += 1.0 # Assign a high score if runtime is zero (or handle as appropriate)
        
        return score

    def judge(self, prompts, completions, shuffle_order=False):
        better_indices = []
        # Ensure we have pairs of completions for each prompt
        if len(completions) % 2 != 0:
             raise ValueError("Completions must be provided in pairs.")
        
        num_prompts = len(prompts)
        num_pairs = len(completions) // 2

        if num_prompts != num_pairs:
             # This might indicate an issue with how data is fed or structured
             print(f"Warning: Number of prompts ({num_prompts}) does not match number of completion pairs ({num_pairs}). Processing based on pairs.")
             # Decide how to handle mismatch: error out or process min(num_prompts, num_pairs)
             # For now, let's process based on the number of pairs available.
             num_prompts = min(num_prompts, num_pairs)


        for i in range(num_pairs):
            prompt = prompts[i] # Assuming one prompt per pair now
            completion1 = completions[2 * i]
            completion2 = completions[2 * i + 1]

            results = []
            for completion in [completion1, completion2]:
                # print(f"Prompt: {prompt}") # Debugging print
                # print(f"Completion: {completion}") # Debugging print
                
                custom_cuda = extract_first_code(completion, ["python", "cpp"])
                # Assuming ref_arch_src comes from the prompt or can be fetched based on it
                # If ref_arch_src is directly in the prompt:
                ref_arch_src = extract_last_code(prompt, ["python", "cpp"]) 
                # If prompt is an ID to fetch the source, adjust accordingly here.
                
                if custom_cuda is None or ref_arch_src is None:
                    print(f"Warning: Could not extract code. Prompt: {prompt[:100]}..., Completion: {completion[:100]}...")
                    # Assign a very low score if code extraction fails
                    results.append(KernelExecResult(compiled=False, correctness=False, runtime=None, error_msg="Code extraction failed"))
                    continue # Skip eval if extraction failed

                try:
                    kernel_exec_result = eval_kernel_against_ref(
                        ref_arch_src, custom_cuda, verbose=False, measure_performance=True, num_correct_trials=5, num_perf_trials=100
                    )
                    results.append(kernel_exec_result)
                except Exception as e:
                    print(f"Error during eval_kernel_against_ref: {e}")
                    # Assign a very low score if evaluation fails unexpectedly
                    results.append(KernelExecResult(compiled=False, correctness=False, runtime=None, error_msg=str(e)))


            if len(results) != 2:
                 print(f"Warning: Expected 2 results for prompt {i}, got {len(results)}. Skipping pair.")
                 # Decide how to handle this: maybe append a default index like 0 or skip?
                 # Skipping might be safer if results are missing due to errors.
                 continue 

            result1, result2 = results
            
            score1 = self._score_completion(result1)
            score2 = self._score_completion(result2)

            if score1 > score2:
                better_indices.append(0)
            elif score2 > score1:
                better_indices.append(1)
            else:
                # Handle ties: randomly choose or default to 0
                better_indices.append(random.choice([0, 1]))
                # better_indices.append(0) # Or default to 0

        return better_indices

