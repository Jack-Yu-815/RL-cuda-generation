from trl import BasePairwiseJudge
import random

# Define a custom judge. reference https://huggingface.co/docs/trl/main/en/judges#define-your-own-judge
class CudaCodeJudge(BasePairwiseJudge):
    def judge(self, prompts, completions, shuffle_order=False):
        return [random.choice([0, 1]) for completion in completions]