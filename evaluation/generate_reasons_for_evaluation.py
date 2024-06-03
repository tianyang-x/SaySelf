from numpy import negative
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_dataset
from datasets import Dataset
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpt_answer_scoring import GPTAnswerScoring
from sentence_transformers import SentenceTransformer
from functools import partial

def preprocess_dataset(examples, args):
    inputs, targets = examples['question'], examples['answer']
    if not args.enable_unsfted_model:
        message_texts = ['[INST]\n' + input.strip() + '[/INST]\n' for input in inputs]
    else:
        message_texts = [f'[INST]\n{input.strip()}\nAfter answering your question, state your reason for your confidence, then represent your confidence in an integer ranging from 1 to 10. The format should be like: \n<step-by-step thinking toward the answer>\nAnswer: <answer to the question>\n Self-reflection: I am not fully confident because of ...\nConfidence: 5[/INST]\n' for input in inputs]
    return {
        'inputs': message_texts,
        'targets': targets,
        'negative_targets': examples['negative_answer']
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="./datasets/mixed_dataset/hotpot_qa.json")
    parser.add_argument('--output_path', type=str, default="./datasets/mixed_dataset/hotpot_qa_test.jsonl")
    # parser.add_argument('--model', type=str, default="./training/output/merged_May13_mistral_lr7e-5_bs128_eg20000")
    parser.add_argument('--model', type=str, default="./training/output/merged_mistral_lr7e-5_bs64_wo_reason")
    parser.add_argument("--num_try",type=int,default=50)
    parser.add_argument('--tokenizer', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples",type=int,default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--enable_unsfted_model", action="store_true", default=False)
    
    args = parser.parse_args()
    output_file = open(args.output_path, 'w')

    dataset = Dataset.from_json(args.dataset_path)[args.start_index:args.start_index + args.num_examples]
    dataset = Dataset.from_dict(dataset)
    preprocess = partial(preprocess_dataset, args=args)
    dataset = dataset.map(preprocess, batched=True, num_proc=16)
    # randomize the dataset
    dataset = dataset.shuffle(seed=42)
    
    model = LLM(model=args.model, tokenizer=args.tokenizer, tensor_parallel_size=args.tensor_parallel_size, seed=args.seed)
    sampling_params = SamplingParams(temperature=1.0, n=args.num_try, top_p=0.95, top_k=50, max_tokens=512)
    
    print("Start inference")
    inputs = dataset['inputs']
    targets = dataset['targets']
    negative_targets = dataset['negative_targets']
    
    # generate by batch
    batch_size = 64
    for i in tqdm(range(0, len(inputs), batch_size)):
        questions = inputs[i:i+batch_size]
        responses = model.generate(inputs[i:i+batch_size], sampling_params=sampling_params)
        # Process the texts seperately
        # Texts follow the format: <COT Chain> Answer: <Answer> Self-reflection: <Self-reflection> Confidence: <Confidence>
        for q, r in zip(questions, responses):
            tt = [response.text for response in r.outputs]
            cot_chain = []
            self_reflection = []
            for text in tt:
                text = text.split("Self-reflection:")
                cot_chain.append(text[0].strip())
                try:
                    text = text[1].split("Confidence:")
                    ref = text[0].strip()
                    text = ref.split("Are you sure you accurately answered the question based on your internal knowledge")
                    self_reflection.append(text[0].strip())
                except Exception:
                    self_reflection.append("N/A")

            reasons = self_reflection
            q = q.replace("[INST]", "").replace("[/INST]", "")
            q = q.replace("After answering your question, state your reason for your confidence, then represent your confidence in an integer ranging from 1 to 10. The format should be like: \n<chain-of-thought of the answer>\nAnswer: <answer to the question>\n Self-reflection: I am not fully confident because of ...\nConfidence: 5", "").strip()
            output_file.write(json.dumps({
                "question": q,
                "reason": reasons,
                "responses": cot_chain,
            }))
            output_file.write("\n")

