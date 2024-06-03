from transformers import StoppingCriteria
import re
from accelerate.logging import get_logger

logger = get_logger(__name__)

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, tokenizer):
        self.target_sequence = target_sequence
        self.tokenizer=tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        find_idx = generated_text.find("<|assistant|>")
        generated_text = generated_text[find_idx:]
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self
        
def parse_response(response):
    # find the last "I am" in sentence
    idx = response.rfind("I am")
    answer, confidence = response[:idx], response[idx:]
    nums = re.findall(r'\d+', confidence)
    if len(nums) != 1:
        return answer, 0
    return answer, float(nums[0])

def parse_response_new(response, args=None):
    if args and args.r_tuning:
        if 'unsure' in response:
            return response, '', 0
        else:
            return response, '', 10
    self_reflection_idx = response.find("Self-reflection: ")
    confidence_idx = response.find("Confidence: ")
    self_reflection_str = response[self_reflection_idx + len("Self-reflection: "):confidence_idx]
    confidence_str = response[confidence_idx + len("Confidence: "):]
    if self_reflection_str == "" or confidence_str == "":
        return response, self_reflection_str, 0
    confidence = re.findall(r'\d+', confidence_str)
    if len(confidence) != 1:
        return response, self_reflection_str, 0
    return response, self_reflection_str, float(confidence[0])

def calculate_reward(correctness, confidence):
    # correctness, confidence: 0~1
    if (not (0 <= correctness <= 10)) or (not (0 <= confidence <= 10)):
        logger.warning(f"Invalid correctness or confidence: {correctness}, {confidence}")
    if not (0 <= correctness <= 10):
        correctness = max(0, min(1, correctness))
    if not (0 <= confidence <= 10):
        confidence = max(0, min(1, confidence))
    confidence /= 10
    correctness /= 10
    ret = 2 * (- abs(correctness - confidence) ** 2 + 0.5)
    return float(ret)
    # if correctness >= 5:
    #     ret = confidence
    # else:
    #     ret = -confidence
    # return ret
