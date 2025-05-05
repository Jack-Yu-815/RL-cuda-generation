from trl import BasePairwiseJudge
import random
from KernelBench.src.eval import eval_kernel_against_ref, KernelExecResult
from KernelBench.src.utils import extract_first_code, extract_last_code, set_gpu_arch
from openai import OpenAI
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import traceback

load_dotenv()

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

    def __init__(self, batch_eval: bool, **kwargs):
        super().__init__(**kwargs)
        self.batch_eval = batch_eval
    
    def judge(self, prompts, completions, shuffle_order=False):
        better_indices = []

        if self.batch_eval:
            batch_prompts = []
            batch_completions = []
            for i in range(len(prompts)):
                for j in range(len(completions[i])):
                    batch_prompts.append(prompts[i])
                    batch_completions.append(completions[i][j])
            
            batch_result = batch_eval(list(zip(batch_prompts, batch_completions)), timeout=150)


        for p_idx in range(len(prompts)):
            prompt = prompts[p_idx]
            
            if not self.batch_eval:
                results = []
                for completion in completions[p_idx]:
                    # 1. Check if the prompt is valid
                    print(f"Completion: {completion}")
                    # if not isinstance(prompt, str) or not isinstance(completion, str):
                    #     raise ValueError("Prompt and completion must be strings")
                    custom_cuda = extract_first_code_relaxed(completion, ["python", "cpp"])
                    ref_arch_src = extract_last_code(prompt, ["python", "cpp"])
                    
                    # check LLM is able to generate custom CUDA code
                    assert custom_cuda is not None, "Custom CUDA code generation failed"
                    set_gpu_arch(["Ada"])
                    try:
                        kernel_exec_result: KernelExecResult = eval_kernel_against_ref(
                            ref_arch_src, custom_cuda, verbose=False, measure_performance=True, num_correct_trials=5, num_perf_trials=10
                        )
                    except Exception as e:
                        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        with open("error.log", "a") as f:
                            f.write(f"Error during eval_kernel_against_ref: {e}\n{tb_str}\n\n\n")
                        kernel_exec_result = KernelExecResult()

                    if kernel_exec_result is None:
                        kernel_exec_result = KernelExecResult()

                    results.append(kernel_exec_result)
                
                result1, result2 = results
            else:
                result1, result2 = batch_result[p_idx*2], batch_result[p_idx*2+1]
            # 2. TODO: FIXME!! Compare the results
            winner = None
            if result1.compiled and result2.compiled:
                if result1.correctness and result2.correctness:
                    winner = 0 if result1.runtime < result2.runtime else 1
                elif result1.correctness ^ result2.correctness:
                    winner = 0 if result1.correctness else 1
            elif result1.compiled ^ result2.compiled:
                winner = 0 if result1.compiled else 1
            better_indices.append(winner)
            
        # filter indices of better_indices that are None, then, batch process gpt_judge
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_index = {executor.submit(gpt_judge, completions[i][0], completions[i][1]): i for i, winner in filter(lambda tpl: tpl[1] is None, enumerate(better_indices))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                winner = future.result()
                print(f"Winner for index {index}: {winner}, {type(winner)}")
                better_indices[index] = winner
        
        assert all([winner in [0, 1] for winner in better_indices]), "All indices should be filled with either 0 or 1"
        assert len(better_indices) == len(prompts), f"Better indices length {len(better_indices)} should be equal to prompts length {len(prompts)}"
        print(f"Better indices: {better_indices}")
        return better_indices



def batch_eval(
    total_work: list[tuple[str, str]],
    timeout: int,  # seconds
):
    """
    Batch evaluation across multiple GPUs, do batch_size of work one on each GPU all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    # construct a list of work args
    batch_size = torch.cuda.device_count()
    set_gpu_arch(["Ada"])

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {batch_size} GPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of GPUs {batch_size}"

            mp.set_start_method('spawn', force=True)
            with mp.Pool(batch_size) as pool:

                work_args = [
                    {
                        "original_model_src": ref_src,
                        "custom_model_src": custom_src,
                        "seed_num": 42,
                        "num_correct_trials": 1,
                        "num_perf_trials": 10,
                        "verbose": False,
                        "measure_performance": True,
                        "device": torch.device(f"cuda:{i%batch_size}"),
                    } for i, (ref_src, custom_src) in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(eval_kernel_against_ref, kwds=work_arg)
                    )
            
                # Collect results with a batch timeout
                results = []
                batch_timeout = timeout
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id = curr_work_batch[i]

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((problem_id, sample_id, result))
                        
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        results.append(KernelExecResult())
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append(KernelExecResult())

                end_time = time.time()


                pbar.update(len(curr_work_batch))
    
    return results


# The custom Gula code may not follow the markdown code format. So, if there's only one code quotation mark, you should also be able to recognize that. 
def extract_first_code_relaxed(text, languages):
    """
    Extract the first code block from the text.
    """
    # count the number of "```" appeared
    count = text.count("```")

    if count == 0:
        return text
    elif count == 1:
        text += "```"
        return extract_first_code(text, languages)
    else:
        return extract_first_code(text, languages)


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def gpt_judge(candidate_1, candidate_2):
    """
    Use GPT to judge the two candidates.
    """
    try:
        messages = [
            {"role": "user", "content": f"""You are a CUDA expert who knows all about writing performant kernels. The following two CUDA program both can't compile or are both return numerically incorrect result. Yet, I need you to pick a relative winner that is closer to a correct and performant solution. Candidate 1:\n```python\n{candidate_1}\n```\n\nCandidate 2:\n```python\n{candidate_2}\n```\n\nFirst, explain your reasoning. Finally, pick a winner by writing a JSON object in the form: ```json\n{{"winner": 1}}\n``` or ```json\n{{"winner": 1}}\n```"""},
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        content = response.choices[0].message.content

        # load the JSON object
        dict_str = extract_last_code(content, ["json"])
        if dict_str is None:
            return random.randint(0, 1)
        else:
            result = json.loads(dict_str)
            return result["winner"] - 1  # adjust range from 1-2 to 0-1
    except Exception as e:
        print(f"Error during gpt_judge execution: {e}. returned random result.")
        return random.randint(0, 1)
