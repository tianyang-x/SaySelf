#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import random
import datasets
from datasets import Dataset
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from deepspeed import get_accelerator
import json
import transformers
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpt_answer_scoring import GPTAnswerScoring
from utils.utils import parse_response_new, calculate_reward
from transformers import StoppingCriteria
from torchmetrics import AUROC
import re

logger = get_logger(__name__)

def encode_with_json_format(example, tokenizer, max_seq_length, args, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    There should only be one assistant message in each example.
    We do not concatenate the assistant messages. 
    '''
    example_text = "[INST]" + example["question"].strip() + "[/INST]\n"
    answer_text = example["answer"].strip()
    if 'negative_answer' in example:
        negative_answer_text = example["negative_answer"]
    else:
        negative_answer_text = None
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    # tokenizer.padding_side = "left"
    if args.explicitly_state_confidence:
        example_text = f'[INST]\nPlease answer the following question step by step. After you have answered it, START A NEW LINE \
            that says \"Confidence: \" and state a number from 1 to 10 that you are confident with your answer. For example: Confidence: 10. \n{example_text.strip()}\n[/INST]\n'
    # if args.r_tuning:
    #     example_text = f'[INST]\n{example_text.strip()} Are you sure you accurately answered the question based on your internal knowledge? [/INST]\n'

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'best_answer': answer_text,
        'question': example["question"].strip(),
        'correct_answers': [answer_text],
        "negative_answers": negative_answer_text.split("/") if negative_answer_text is not None else None,
    }

def encode_with_files_format(example, tokenizer, max_seq_length, args, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    There should only be one assistant message in each example.
    We do not concatenate the assistant messages. 
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        answer_text = ""
        if args.explicitly_state_confidence:
            messages[0]['content'] = 'Please answer the following question step by step. After you have answered it, START A NEW LINE \
            that says \"Confidence: \" and state a number from 1 to 10 that you are confident with your answer. For example: Confidence: 10.' + messages[0]['content']
        # if args.r_tuning:
        #     messages[0]['content'] = messages[0]['content'] + ' Are you sure you accurately answered the question based on your internal knowledge?'
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                message_text += "[INST]\n" + message["content"].strip() + "[/INST]\n"
            elif message["role"] == "user":
                message_text += "[INST]\n" + message["content"].strip() + "[/INST]\n"
            elif message["role"] == "assistant" and idx != len(messages) - 1:
                message_text += "\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            elif message["role"] == "assistant" and idx == len(messages) - 1:
                answer_text = message["content"].strip()
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text, answer_text
        
    example_text, answer_text = _concat_messages(messages)
    i_am_idx = answer_text.rfind("I am")
    if i_am_idx != -1:
        answer_text = answer_text[:i_am_idx]

    if add_bos:
        example_text = tokenizer.bos_token + example_text
    # tokenizer.padding_side = "left"
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'best_answer': answer_text,
        'question': messages[-2]['content'].strip(),
        'correct_answers': [answer_text]
    }


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        find_idx = generated_text.find("[/INST]")
        generated_text = generated_text[find_idx:]
        # Check if words like "Confidence: \d" have been generated
        if re.search(r"Confidence: \d+", generated_text):
            return True
        if re.search(r"I am sure", generated_text) or re.search(r"I am unsure", generated_text):
            return True

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class MyDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, batch, *args, **kwargs):
        batch2 = [{'input_ids': d['input_ids']} for d in batch]
        best_answer, correct_answers, question = [b['best_answer'] for b in batch], [b['correct_answers'] for b in batch], [b['question'] for b in batch]
        b2 = super().__call__(batch2)
        b2['best_answer'] = best_answer
        b2['correct_answers'] = correct_answers
        b2['question'] = question
        return b2

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--train_micro_batch_size_per_gpu",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        '--num_eval_examples',
        type=int,
        default=20000,
        help='Number of evaluation examples to use. Default is 20000.',
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start index of the evaluation examples to use. Default is 0.',
    )
    parser.add_argument(
        '--explicitly_state_confidence',
        action='store_true',
    )
    parser.add_argument(
        '--r_tuning',
        action='store_true',
    )
    parser.add_argument(
        '--train_file',
        type=str
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    return args

def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False, args=None):
    '''
    Here we assume that the dataset follows the format of "truthful_qa" dataset in Huggingface Datasets library.
    '''
    question, best_answer, correct_answers = example['question'], example['best_answer'], example['correct_answers']
    example = f'[INST]\n{question.strip()}\n[/INST]\n'
    if args.explicitly_state_confidence:
        example = f'[INST]\nPlease answer the following question step by step. After you have answered it, START A NEW LINE \
            that says \"Confidence: \" and state a number from 1 to 10 that you are confident with your answer. For example: Confidence: 10. \n{question.strip()}\n[/INST]\n'
    # if args.r_tuning:
    #     example = f'[INST]\n{question.strip()} Are you sure you accurately answered the question based on your internal knowledge?\n[/INST]\n'

    if add_bos:
        example = tokenizer.bos_token + example
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_example = tokenizer(example, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    # attention_masks = tokenized_example.attention_mask
    return {
        'input_ids': input_ids.flatten(),
        # 'attention_masks': attention_masks.flatten(),
        'question': question,
        'best_answer': best_answer,
        'correct_answers': correct_answers,
    }

def encode_with_strategy_format(example, tokenizer, max_seq_length, add_bos=False, args=None):
    '''
    Here we assume that the dataset follows the format of "truthful_qa" dataset in Huggingface Datasets library.
    '''
    question, best_answer = example['question'], example['answer']
    best_answer = 'False' if not best_answer else "True"
    example = f'[INST]\n{question.strip()}\n[/INST]\n'
    if args.explicitly_state_confidence:
        example = f'[INST]\nPlease answer the following question step by step. After you have answered it, START A NEW LINE \
            that says \"Confidence: \" and state a number from 1 to 10 that you are confident with your answer. For example: Confidence: 10. \n{question.strip()}\n[/INST]\n'
    # if args.r_tuning:
    #     example = f'[INST]\n{question.strip()} Are you sure you accurately answered the question based on your internal knowledge?\n[/INST]\n'

    if add_bos:
        example = tokenizer.bos_token + example
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_example = tokenizer(example, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    # attention_masks = tokenized_example.attention_mask
    return {
        'input_ids': input_ids.flatten(),
        # 'attention_masks': attention_masks.flatten(),
        'question': question,
        'best_answer': best_answer,
        'correct_answers': [best_answer],
    }

def main():
    args = parse_args()
    output_reasons = []

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        try:
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split="validation",
            )
        except Exception as e:
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split="test",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        try:
            raw_datasets_dict = json.load(open(args.train_file))
            raw_datasets = datasets.Dataset.from_dict(raw_datasets_dict)
            # filter out examples with too long messages (Currently set to 400 words)
            raw_datasets = raw_datasets.filter(lambda example: sum([len(d['content'].split()) for d in example['messages']]) < 200)
        except Exception as e:
            raw_datasets_dict = json.load(open(args.train_file))
            new_dict = {"question": [], "answer": [], "negative_answer": []}
            for d in raw_datasets_dict:
                new_dict["question"].append(d["question"])
                new_dict["answer"].append(d["answer"])
                new_dict["negative_answer"].append(d["negative_answer"])
            raw_datasets = datasets.Dataset.from_dict(new_dict)

    raw_datasets = Dataset.from_dict(raw_datasets[args.start_index:args.start_index + args.num_eval_examples])
    # Load pretrained model and tokenizer
    if args.config_name:
        config = None
    elif args.model_name_or_path:
        config = None
    elif args.peft_model_name_or_path:
        config = None
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=args.trust_remote_code, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=not args.use_slow_tokenizer)
    elif args.peft_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.peft_model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        )
    elif args.peft_model_name_or_path:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.peft_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        )
    else:
        raise ValueError("You are instantiating a new model from scratch. This is not supported by this script.")

    answer_scorer = GPTAnswerScoring()
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": 256,
        "top_k": 50,
        # "top_p": 0.9,
        "temperature": 0.7,
        "num_return_sequences": 1,
    }

    # Preprocessing the datasets.
    if "messages" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_files_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
            args=args,
        )
    elif "facts" in raw_datasets.column_names and "term" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_strategy_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
            args=args,
        )
    elif "question" in raw_datasets.column_names and "answer" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_json_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
            args=args,
        )
    else:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
            args=args,
        )
    
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")

    print(f"lm dataset length: {len(lm_datasets)}")
    test_dataset = lm_datasets
    print(f"dataset: {test_dataset}")
    print(f"dataset length: {len(test_dataset)}")
    
    if AcceleratorState().is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    # DataLoaders creation:
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.train_micro_batch_size_per_gpu,
        shuffle=False,
        collate_fn=MyDataCollator(tokenizer, padding=True),
    )

    model.eval()
    model, tokenizer, test_dataset, test_dataloader = accelerator.prepare(model, tokenizer, test_dataset, test_dataloader)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        accelerator.init_trackers("reinforcement_eval", experiment_config)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_micro_batch_size_per_gpu}")
    # Only show the progress bar once on each machine.
    completed_steps = 0

    active_dataloader = test_dataloader
    ece = []
    accuracy = []
    conf_pos = []
    conf_neg = []
    all_confidences = []
    all_auroc_target = []
    for step, batch in tqdm(enumerate(active_dataloader), disable=not accelerator.is_local_main_process, desc="Evaluation", total=len(active_dataloader)):
        # do evaluation on test sets
        # try:
        input_ids, masks = batch['input_ids'], batch['attention_mask']
        question, best_answer, correct_answers = batch['question'], batch['best_answer'], batch['correct_answers']
        
        # get response from model
        response_tensors = model.generate(input_ids, attention_mask=masks, **generation_kwargs,
                                        stopping_criteria=MyStoppingCriteria(tokenizer))
        response = [tokenizer.decode(r.squeeze(), skip_special_tokens=True).replace('<pad>', '') for r in response_tensors]
        response = [response[response.find('[/INST]'):] for response in response]
        
        # Check each response: if a response is nonsensical or doesn't contain a percent sign, retry the generation by telling them I am ... sure
        to_retry_idx = []
        retried_old_length = {}
        for x, r in enumerate(response):
            if len(r) == 0:
                to_retry_idx.append(x)
                logger.warning(f"Response {x} is empty. Retrying.")
                continue
            if not args.r_tuning:
                percent_idx = r.find('Confidence:')
                if not ('Confidence:' in r):
                    to_retry_idx.append(x)
                    logger.warning(f"Response {x} does not include confidence. Retrying.")
            else:
                if not ('I am sure' in r or 'I am unsure' in r):
                    to_retry_idx.append(x)
                    logger.warning(f"Response {x} does not include confidence. Retrying.")

        for i in to_retry_idx:
            num_tries = 0
            while num_tries < 3:
                to_retry_response = response[i]
                retry_input_ids = tokenizer(to_retry_response, return_tensors='pt').to(model.device)
                new_response_tensors = model.generate(**retry_input_ids, **generation_kwargs,
                                        stopping_criteria=MyStoppingCriteria(tokenizer))
                new_response = tokenizer.decode(new_response_tensors[0], skip_special_tokens=True).replace('<pad>', '')
                new_response = new_response[new_response.find('[/INST]'):]
                # response_tensors[i] = new_response_tensors[0]
                response[i] = new_response
                if not args.r_tuning:
                    if 'Confidence:' in new_response:
                        logger.info(f"Retried response {i} successfully. New response: {new_response}")
                        break
                else:
                    if 'I am sure' in new_response or 'I am unsure' in new_response:
                        logger.info(f"Retried response {i} successfully. New response: {new_response}")
                        break
                num_tries += 1
            if 'Confidence:' in new_response or 'I am sure' in new_response or 'I am unsure' in new_response:
                continue
            
            # if we still can't get a response with a percent sign, we add "Confidence:" to input
            logger.info(f"Retried response {i}: trying to add Confidence: to input.")
            to_retry_response = response[i]
            to_retry_resp_len = len(response_tensors[i])
            if not args.r_tuning:
                retry_input_ids = tokenizer(to_retry_response + '. Confidence:', return_tensors='pt').to(model.device)
            else:
                retry_input_ids = tokenizer(to_retry_response + '.  Are you sure you accurately answered the question based on your internal knowledge?', return_tensors='pt').to(model.device)
            new_response_tensors = model.generate(**retry_input_ids, **generation_kwargs,
                                    stopping_criteria=MyStoppingCriteria(tokenizer))
            new_response = tokenizer.decode(new_response_tensors[0], skip_special_tokens=True).replace('<pad>', '')
            new_response = new_response[new_response.find('[/INST]'):]
            # response_tensors[i] = new_response_tensors[0]
            response[i] = new_response
            if 'Confidence:' in new_response or 'I am sure' in new_response or 'I am unsure' in new_response:
                logger.info(f"Retried response {i} successfully. New response: {new_response}")
            else:
                logger.warning(f"Retried response {i} unsuccessfully. New response: {new_response}")
                retried_old_length[i] = to_retry_resp_len
        
        if step % 10 == 0:
            logger.info(f"Example responses: {response[:5]}")
        
        parsed = [parse_response_new(r, args) for r in response]
        answers, _, confidences = zip(*parsed)
        confidences = [c for c in confidences]
        
        # compute correctness
        correctness_ret = []
        
        correctness_ret = [answer_scorer.score(q, a, b) for q, a, b, c in zip(question, answers, best_answer, correct_answers)]
        correctness = correctness_ret
        rewards = [calculate_reward(c, conf) for c, conf in zip(correctness, confidences)]
        
        output_reasons.extend([{
            "question": q,
            "best_answer": b,
            "correct_answers": c,
            "response": e,
            "correctness": r,
        }
        for q, b, c, r, e in zip(question, best_answer, correct_answers, correctness_ret, response)])

        # calculate the ECE (expected calibration error)
        partial_ece = [abs(c - r) / 10 for c, r in zip(confidences, correctness)]
        ece.extend(partial_ece)
        partial_ece_mean = sum(partial_ece) / len(partial_ece)
        accelerator.log({"test/ece": partial_ece_mean})
        logger.info(f"ECE: {partial_ece_mean} at step {completed_steps}")
        cumulative_ece = sum(ece) / len(ece)
        accelerator.log({"test/cumulative_ece": cumulative_ece})
        logger.info(f"Cumulative ECE: {cumulative_ece} at step {completed_steps}")

        # calculate the accuracy and conf_pos/conf_neg
        partial_accuracy = [c / 10 for c in correctness]
        accuracy.extend(partial_accuracy)
        partial_accuracy_mean = sum(partial_accuracy) / len(partial_accuracy)
        accelerator.log({"test/accuracy": partial_accuracy_mean})
        logger.info(f"Accuracy: {partial_accuracy_mean} at step {completed_steps}")
        cumulative_accuracy = sum(accuracy) / len(accuracy)
        accelerator.log({"test/cumulative_accuracy": cumulative_accuracy})
        logger.info(f"Cumulative accuracy: {cumulative_accuracy} at step {completed_steps}")

        # calculate conf_pos and conf_neg
        # partial_conf_pos = [c * r for c, r in zip(confidences, correctness)]
        # partial_conf_neg = [c * (1 - r) for c, r in zip(confidences, correctness)]
        partial_conf_pos = [c / 10 for c, r in zip(confidences, correctness) if r >= 7]
        partial_conf_neg = [c / 10 for c, r in zip(confidences, correctness) if r <= 3]
        if len(partial_conf_pos) == 0:
            partial_conf_pos = [0]
        if len(partial_conf_neg) == 0:
            partial_conf_neg = [0]
        conf_pos.extend(partial_conf_pos)
        conf_neg.extend(partial_conf_neg)
        partial_conf_pos_mean = sum(partial_conf_pos) / len(partial_conf_pos) if len(partial_conf_pos) > 0 else 0
        partial_conf_neg_mean = sum(partial_conf_neg) / len(partial_conf_neg) if len(partial_conf_neg) > 0 else 0
        accelerator.log({"test/conf_pos": partial_conf_pos_mean})
        accelerator.log({"test/conf_neg": partial_conf_neg_mean})
        logger.info(f"Conf_pos: {partial_conf_pos_mean} at step {completed_steps}")
        logger.info(f"Conf_neg: {partial_conf_neg_mean} at step {completed_steps}")
        cumulative_conf_pos = sum(conf_pos) / len(conf_pos) if len(conf_pos) > 0 else 0
        cumulative_conf_neg = sum(conf_neg) / len(conf_neg) if len(conf_neg) > 0 else 0
        accelerator.log({"test/cumulative_conf_pos": cumulative_conf_pos})
        accelerator.log({"test/cumulative_conf_neg": cumulative_conf_neg})
        logger.info(f"Cumulative conf_pos: {cumulative_conf_pos} at step {completed_steps}")
        logger.info(f"Cumulative conf_neg: {cumulative_conf_neg} at step {completed_steps}")

        # calculate AUROC
        all_confidences.extend([c / 10 for c in confidences])
        all_auroc_target.extend([1 if c >= 7 else 0 for c in correctness])
        auroc = AUROC(task="binary")
        auroc_value = auroc(torch.tensor(all_confidences), torch.tensor(all_auroc_target))
        accelerator.log({"test/auroc": auroc_value})
        logger.info(f"AUROC: {auroc_value} at step {completed_steps}")
    
        # except RuntimeError as e:
        #     logger.error(f"Error in step {step}: {e}")
        #     logger.error(f"When processing batch: {batch}")
        #     if 'CUDA out of memory' in str(e):
        #         logger.warn("Emptying cache and trying again.")
        #         for p in model.parameters():
        #             if p.grad is not None:
        #                 del p.grad
        #         get_accelerator().empty_cache()
        #         continue
        #     else:
        #         raise e

        get_accelerator().empty_cache()

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        # Output reasons for the responses
        with open(os.path.join(args.output_dir, "output_reasons.json"), "w") as f:
            json.dump(output_reasons, f)
            

if __name__ == "__main__":
    main()
