#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
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
import deepspeed
from deepspeed import get_accelerator
import json
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import re
from peft import AutoPeftModelForCausalLM, get_peft_model, PeftModel
from transformers import MistralForCausalLM, StoppingCriteria
import transformers
import traceback
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpt_answer_scoring import GPTAnswerScoring
from utils.utils import MyStoppingCriteria, parse_response, calculate_reward, parse_response_new
from huggingface_hub import login

logger = get_logger(__name__)

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

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class MyDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, batch, *args, **kwargs):
        batch2 = [{'input_ids': d['input_ids'], 'attention_mask': d['attention_mask']} for d in batch]
        answer, question = [b['answer'] for b in batch], [b['question'] for b in batch]
        b2 = super().__call__(batch2)
        b2['answer'] = answer
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
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
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
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
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
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
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
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
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
        '--num_train_examples',
        type=int,
        default=20000,
        help='Number of training examples to use. Default is 20000.',
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start index of the training examples to use. Default is 0.',
    )
    parser.add_argument(
        '--num_retries',
        type=int,
        default=3,
        help='Number of retries for tokenization. Default is 3.',
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
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
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                message_text += "[INST]\n" + message["content"].strip() + "[/INST]\n"
            elif message["role"] == "user":
                message_text += "[INST]\n" + message["content"].strip() + "[/INST]\n"
            elif message["role"] == "assistant" and idx != len(messages) - 1:
                message_text += "\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            elif message["role"] == "assistant" and idx == len(messages) - 1:
                message_text += "\n"
                answer_text = message["content"].strip()
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text, answer_text
    example_text, answer_text = _concat_messages(messages)

    if add_bos:
        example_text = tokenizer.bos_token + example_text
    # tokenizer.padding_side = "left"
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'answer': answer_text,
        'attention_mask': attention_mask.flatten(),
        'question': messages[-2]['content'].strip(),
    }


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    # create output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False
        )

def main():
    try:
        args = parse_args()
        output_reasons = []

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {
            # "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }

        if args.with_tracking:
            accelerator_log_kwargs["log_with"] = args.report_to
            # accelerator_log_kwargs["project_dir"] = args.output_dir

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
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
            )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            raw_datasets_dict = json.load(open(args.train_file))
            raw_datasets = datasets.Dataset.from_dict(raw_datasets_dict)
            raw_datasets = raw_datasets.train_test_split(test_size=0.01, seed=args.seed)
            # filter out examples with too long messages (Currently set to 400 words)
            print(f"Currently, there are {len(raw_datasets['train'])} examples in the training set.")
            raw_datasets['train'] = raw_datasets['train'].filter(lambda example: sum([len(d['content'].split()) for d in example['messages']]) < 100)
            raw_datasets['test'] = raw_datasets['test'].filter(lambda example: sum([len(d['content'].split()) for d in example['messages']]) < 100)
            print(f"After filtering, there are {len(raw_datasets['train'])} examples in the training set.")
            raw_datasets['train'] = datasets.Dataset.from_dict(raw_datasets['train'][:args.num_train_examples])

        # read ground truth dataset
        # ground_truth_dataset = []
        # if args.ground_truth_dataset_path is not None:
        #     for line in open(args.ground_truth_dataset_path):
        #         ground_truth_dataset.append(json.loads(line))
        
        raw_datasets_train = Dataset.from_dict(raw_datasets['train'][args.start_index:args.start_index + args.num_train_examples])
        # Random choose: size of test set is 1% of the training set
        raw_datasets_test = Dataset.from_dict(raw_datasets['test'][:len(raw_datasets['train']) // 100])
        raw_datasets = datasets.DatasetDict({'train': raw_datasets_train, 'test': raw_datasets_test})
        # Load pretrained model and tokenizer
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
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
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                attn_implementation="flash_attention_2" if args.use_flash_attn else None,
            )
        elif args.peft_model_name_or_path:
            sft_model = AutoPeftModelForCausalLM.from_pretrained(args.peft_model_name_or_path, is_trainable=True)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                sft_model,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                is_trainable=True,
                attn_implementation="flash_attention_2" if args.use_flash_attn else None,
            )
        else:
            print("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
        
        ppo_config = PPOConfig(
            model_name=args.model_name_or_path,
            query_dataset=args.dataset_name if args.dataset_name is not None else "custom",
            reward_model=GPTAnswerScoring(),
            tracker_project_name="score_reinforcement",
            learning_rate=args.learning_rate,
            # batch_size=args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
            batch_size=args.train_micro_batch_size_per_gpu,
            # gradient_accumulation_steps=args.gradient_accumulation_steps,
            # world_size=get_accelerator().num_processes,
            ppo_epochs=args.num_train_epochs,
            max_grad_norm=0,
            optimize_device_cache=True,
            use_score_scaling=True,
            use_score_norm=True,
            whiten_rewards=False,
            **accelerator_log_kwargs,
            accelerator_kwargs={"step_scheduler_with_optimizer": False,},
        )
        
        answer_scorer = GPTAnswerScoring()
        
        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": 150,
            "top_k": 50,
            # "top_p": 0.9,
            "temperature": 0.7,
            "num_return_sequences": 1,
        }

        # no default pad token for llama!
        # here we add all special tokens again, because the default ones are not in the special_tokens_map
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast) or isinstance(model, MistralForCausalLM):
            num_added_tokens = tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
        elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
            num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

        if args.use_lora:
            logger.info("Initializing LORA model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=args.lora_rank, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        if args.peft_model_name_or_path:
            model.pretrained_model.print_trainable_parameters() 

        # Preprocessing the datasets.
        if "messages" in raw_datasets["train"].column_names:
            encode_function = partial(
                encode_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                add_bos=args.add_bos,
            )
        else:
            raise ValueError("Dataset format error: 'messages' field not found.")
        
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")

        print(f"lm dataset length: {len(lm_datasets)}")
        train_dataset = lm_datasets["train"]
        test_dataset = lm_datasets["test"]
        print(f"Training dataset: {train_dataset}")
        print(f"Training dataset length: {len(train_dataset)}")
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        ppo_trainer = PPOTrainer(
            model=model,
            dataset=train_dataset,
            tokenizer=tokenizer,
            config=ppo_config,
            optimizer=optimizer,
        )
        logger.info("PPOTrainer initialized.")
        
        if AcceleratorState().is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_micro_batch_size_per_gpu,
            shuffle=True,
            collate_fn=MyDataCollator(tokenizer),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.train_micro_batch_size_per_gpu,
            shuffle=False,
            collate_fn=MyDataCollator(tokenizer),
        )

        train_dataset, train_dataloader = ppo_trainer.accelerator.prepare(train_dataset, train_dataloader)
        test_dataset, test_dataloader = ppo_trainer.accelerator.prepare(test_dataset, test_dataloader)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        # Create the learning rate scheduler.
        # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
        # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
        # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
        # number of updates in the end matches the num_training_steps here.
        # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
        # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
        num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * ppo_trainer.accelerator.num_processes
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            ppo_trainer.accelerator.init_trackers("open_instruct", experiment_config)

        # Train!
        total_batch_size = args.train_micro_batch_size_per_gpu * ppo_trainer.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_micro_batch_size_per_gpu}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not ppo_trainer.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            ppo_trainer.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            ppo_trainer.accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(checkpoint_path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
                )
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        last_lowest_ece = 1
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
            ):
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = ppo_trainer.accelerator.skip_first_batches(
                    train_dataloader, resume_step
                )
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                try:
                    with ppo_trainer.accelerator.accumulate(model):
                        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                        labels = batch['answer']
                        questions = batch['question']
                        
                        # get response from model
                        input_ids = list(input_ids)
                        # change by attention mask
                        for i, mask in enumerate(attention_mask):
                            input_ids[i] = input_ids[i][mask == 1]
                        response_tensors = ppo_trainer.generate(input_ids, **generation_kwargs,
                                                                stopping_criteria=MyStoppingCriteria(tokenizer))
                        response = [tokenizer.decode(r.squeeze(), skip_special_tokens=True).replace('<pad>', '') for r in response_tensors]
                        
                        # Check each response: if a response is nonsensical or doesn't contain a percent sign, retry the generation by telling them I am ... sure
                        to_retry_idx = []
                        retried_old_length = {}
                        for x, r in enumerate(response):
                            if len(r) == 0:
                                to_retry_idx.append(x)
                                logger.warning(f"Response {x} is empty. Retrying.")
                                continue
                            percent_idx = r.find('Confidence:')
                            if not ('Confidence:' in r):
                                to_retry_idx.append(x)
                                logger.warning(f"Response {x} does not include confidence. Retrying.")
                        for i in to_retry_idx:
                            num_tries = 0
                            while num_tries < args.num_retries:
                                to_retry_response = response[i]
                                retry_input_ids = tokenizer(to_retry_response, return_tensors='pt')['input_ids']
                                new_response_tensors = ppo_trainer.generate([retry_input_ids[0].squeeze(0)], **generation_kwargs,
                                                        stopping_criteria=MyStoppingCriteria(tokenizer))
                                new_response = tokenizer.decode(new_response_tensors[0], skip_special_tokens=True).replace('<pad>', '')
                                response_tensors[i] = new_response_tensors[0]
                                response[i] = new_response
                                if 'Confidence:' in new_response:
                                    logger.info(f"Retried response {i} successfully. New response: {new_response}")
                                    break
                                num_tries += 1
                            if 'Confidence:' in new_response:
                                continue
                            # if we still can't get a response with a percent sign, we add "Confidence:" to input
                            logger.info(f"Retried response {i}: trying to add Confidence: to input.")
                            to_retry_response = response[i]
                            to_retry_resp_len = len(response_tensors[i])
                            retry_input_ids = tokenizer(to_retry_response + '. Confidence:', return_tensors='pt')['input_ids']
                            new_response_tensors = ppo_trainer.generate([retry_input_ids[0].squeeze(0)], **generation_kwargs,
                                                    stopping_criteria=MyStoppingCriteria(tokenizer))
                            new_response = tokenizer.decode(new_response_tensors[0], skip_special_tokens=True).replace('<pad>', '')
                            response_tensors[i] = new_response_tensors[0]
                            response[i] = new_response
                            percent_idx = new_response.find('Confidence:')
                            if 'Confidence:' in new_response:
                                logger.info(f"Retried response {i} successfully. New response: {new_response}")
                            else:
                                logger.warning(f"Retried response {i} unsuccessfully. New response: {new_response}")
                                retried_old_length[i] = to_retry_resp_len
                        
                        if step % 10 == 0:
                            logger.info(f"Example responses: {response[:5]}")
                            
                        parsed = [parse_response_new(r) for r in response]
                        answers, self_reflections, confidences = zip(*parsed)
                        
                        # compute correctness
                        # check if correct confidence exists in the file
                        correctness_ret = []
                        for q, a, l in zip(questions, answers, labels):
                            if l.lower().strip() in a.lower().strip():
                                correctness_ret.append(10)
                            else:
                                correctness_ret.append(0)
                        # correctness_ret = [answer_scorer.score(q, answer, label) for q, answer, label in zip(questions, answers, labels)]
                        correctness = correctness_ret
                        rewards = [calculate_reward(c, conf) for c, conf in zip(correctness, confidences)]
                        reward_tensor = [torch.tensor([r], device=batch['input_ids'].device) for r in rewards]
                        
                        output_reasons.extend([{
                            "question": q,
                            "answer": l,
                            "response": r,
                            "correctness": c,
                        }
                        for q, l, c, r in zip(questions, labels, correctness_ret, response)])
                        log_batch = {
                            'query': batch['question'],
                            'response': batch['answer']
                        }
                        
                        # set response masks for PPOTrainer
                        response_masks = [torch.zeros_like(r).to(r.device) for r in response_tensors]
                        for x, r in enumerate(response_tensors):
                            first_digit_token = None
                            for y in range(r.size(0)):
                                # find "Confidence:"
                                tokens_to_here = tokenizer.decode(r[:y], skip_special_tokens=True)
                                if 'Confidence:' in tokens_to_here:
                                    first_digit_token = y
                                    break
                            if first_digit_token is not None and first_digit_token < r.size(0):
                                response_masks[x][first_digit_token:] = 1
                            else:
                                # In this case, the response is nonsensical and we set the reward to -1
                                reward_tensor[x] = torch.tensor([-1.0], device=batch['input_ids'].device)
                                # set response mask
                                logger.warning(f"Response {x} is nonsensical. Setting reward to -1. Response: {response[x]}")
                                try:
                                    response_masks[x][retried_old_length[x]:] = 1
                                except Exception:
                                    response_masks[x][:] = 1
                        # response_masks = response_masks.bool()
                        
                        if len(input_ids) == 0:
                            logger.warning("All responses have been deleted. Skipping this step.")
                            continue
                        assert len(response_tensors) == len(response_masks), f"Response tensors and masks have different lengths: {len(response_tensors)} vs {len(response_masks)}"
                        for x, r in zip(response_tensors, response_masks):
                            assert x.size(0) == r.size(0), f"Response tensor and mask have different lengths: {x.size(0)} vs {r.size(0)}"
                        # logger.info(f"unmasked response: {response_tensors[0][response_masks[0].bool()]}, {tokenizer.decode(response_tensors[0][response_masks[0].bool()])}")
                        stats = ppo_trainer.step(input_ids, response_tensors, reward_tensor, response_masks)
                        # delete all inf and nan values in stats
                        # stats = {k: v for k, v in stats.items() if not torch.isnan(v) and not torch.isinf(v)}
                        reward_tensor = [torch.nan_to_num(t) for t in reward_tensor]
                        ppo_trainer.log_stats(stats, log_batch, reward_tensor)
                except RuntimeError as e:
                    logger.error(f"Error in step {step}: {e}")
                    logger.error(f"When processing batch: {batch}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    if 'CUDA out of memory' in str(e):
                        logger.warning("Emptying cache and trying again.")
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad
                        get_accelerator().empty_cache()
                        continue
                    else:
                        raise e
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if ppo_trainer.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                        
                    if completed_steps % 10 == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(ppo_trainer.accelerator, model, tokenizer, output_dir, args)

                    if completed_steps >= args.max_train_steps:
                        break
                get_accelerator().empty_cache()
                
                # do evaluation for checkpointing steps
                if completed_steps % 10 == 0:
                    ece = []
                    for step, batch in tqdm(enumerate(test_dataloader), desc="Evaluating", total=len(test_dataloader)):
                        try:
                            with ppo_trainer.accelerator.accumulate(model):
                                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                                labels = batch['answer']
                                questions = batch['question']
                                
                                # get response from model
                                input_ids = list(input_ids)
                                response_tensors = ppo_trainer.generate(input_ids, **generation_kwargs,
                                                                        stopping_criteria=MyStoppingCriteria(tokenizer))
                                response = [tokenizer.decode(r.squeeze()) for r in response_tensors]
                                parsed = [parse_response_new(r) for r in response]
                                answers, _, confidences = zip(*parsed)
                                
                                # compute correctness
                                correctness_ret = [answer_scorer.score(q, answer, label) for q, answer, label in zip(questions, answers, labels)]
                                correctness = [c for c in correctness_ret]
                                rewards = [calculate_reward(c, conf) for c, conf in zip(correctness, confidences)]
                                
                                # calculate the ECE (expected calibration error)
                                ece.extend([abs(c - r) for c, r in zip(confidences, rewards)])
                                
                        except RuntimeError as e:
                            logger.error(f"Error in step {step}: {e}")
                            logger.error(f"When processing batch: {batch}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")
                            if 'CUDA out of memory' in str(e):
                                logger.warning("Emptying cache and trying again.")
                                for p in model.parameters():
                                    if p.grad is not None:
                                        del p.grad
                                get_accelerator().empty_cache()
                                continue
                            else:
                                raise e
                        get_accelerator().empty_cache()
                        
                    ece = sum(ece) / len(ece)
                    ppo_trainer.accelerator.log({"test/ece": ece})
                    logger.info(f"ECE: {ece} at step {completed_steps}")
                    
                    if ece < last_lowest_ece:
                        last_lowest_ece = ece
                        # save model with lowest ECE value
                        output_dir = f"lowest_ece"
                        if args.output_dir is not None:
                            logger.info(f"Saving model with lowest ECE value: {last_lowest_ece}")
                            output_dir = os.path.join(args.output_dir, output_dir)
                            save_with_accelerate(ppo_trainer.accelerator, model, tokenizer, output_dir, args)

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                save_with_accelerate(ppo_trainer.accelerator, model, tokenizer, output_dir, args)
                
        if args.with_tracking:
            ppo_trainer.accelerator.end_training()

        if args.output_dir is not None:
            ppo_trainer.accelerator.wait_for_everyone()
            if ppo_trainer.accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
            save_with_accelerate(ppo_trainer.accelerator, model, tokenizer, args.output_dir, args)
            # Output reasons for the responses
            with open(os.path.join(args.output_dir, "output_reasons.json"), "w") as f:
                json.dump(output_reasons, f)
    except KeyboardInterrupt as e:
        # save model
        logger.info("SIGINT received. Saving model...")
        if args.output_dir is not None:
            output_dir = f"epoch_int"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(ppo_trainer.accelerator, model, tokenizer, output_dir, args)
            # Output reasons for the responses
            with open(os.path.join(args.output_dir, "output_reasons.json"), "w") as f:
                json.dump(output_reasons, f)
        raise e


if __name__ == "__main__":
    main()
