#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    format_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, ModelConfig, ScriptArguments, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


@dataclass
class PPOScriptArguments(ScriptArguments):
    """
    Script arguments for the PPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B",
        metadata={"help": "Path to the model to use."},
    )
    output_dir: str = field(
        default="./output/ppo_qwen",
        metadata={"help": "Output directory for model and checkpoints."},
    )
    wandb_project: str = field(
        default="PPO_Qwen",
        metadata={"help": "WandB project name."},
    )


@dataclass
class PPOConfig(GRPOConfig):
    """PPO configuration for training."""
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate"})
    mini_batch_size: int = field(default=8, metadata={"help": "PPO mini batch size"})
    batch_size: int = field(default=32, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    ppo_epochs: int = field(default=4, metadata={"help": "Number of PPO epochs"})
    max_steps: int = field(default=1000, metadata={"help": "Maximum number of training steps"})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X steps"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize WandB if needed
    init_wandb_training(training_args)

    # Load datasets
    train_dataset = load_dataset(
        "csv", 
        data_files={"train": "InternDay_RLVR/open-r1/data/train.csv"},
        split="train"
    )
    
    valid_dataset = load_dataset(
        "csv", 
        data_files={"valid": "InternDay_RLVR/open-r1/data/validation.csv"},
        split="valid"
    )

    # Initialize tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    
    # If no padding token, pad with eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None),
        device_map="auto",
    )
    
    # Apply PEFT config if specified
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
        model.pretrained_model = model.get_peft_model(peft_config)

    # Get reward functions
    def combine_rewards(responses, query=None, answer=None, **kwargs):
        """Combine multiple reward functions."""
        rewards = []
        
        if "accuracy" in script_args.reward_funcs:
            rewards.append(accuracy_reward(responses, answer, **kwargs))
        
        if "format" in script_args.reward_funcs:
            rewards.append(format_reward(responses, **kwargs))
            
        # Combine rewards with equal weights
        combined_rewards = []
        for i in range(len(responses)):
            total_reward = sum(reward[i] for reward in rewards) / len(rewards)
            combined_rewards.append(float(total_reward))
            
        return combined_rewards

    # Define data preparation function
    def prepare_data(examples):
        formatted_examples = []
        
        for i in range(len(examples["Question"])):
            question = examples["Question"][i]
            choices = examples["Multiple_Choices"][i]
            correct_choice = examples["Correct_Answer_Choice"][i]
            
            # Format input
            prompt = f"Question: {question}\nChoices: {choices}\nPlease provide your answer with explanation."
            
            # Format expected output for reward calculation
            expected_answer = correct_choice
            
            formatted_examples.append({
                "query": prompt,
                "answer": expected_answer
            })
            
        return formatted_examples

    # Prepare datasets
    prepared_train_dataset = prepare_data(train_dataset)
    prepared_valid_dataset = prepare_data(valid_dataset)

    # Initialize PPO Trainer
    ppo_config = PPOConfig(
        learning_rate=training_args.learning_rate,
        batch_size=training_args.batch_size,
        mini_batch_size=training_args.mini_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=training_args.ppo_epochs,
        max_steps=training_args.max_steps,
        output_dir=training_args.output_dir,
        save_steps=training_args.save_steps,
    )
    
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=prepared_train_dataset,
    )

    # Get callbacks
    callbacks = get_callbacks(training_args, model_args)
    
    # Training loop
    for step, batch in enumerate(trainer.dataloader):
        # Check if max steps reached
        if step >= ppo_config.max_steps:
            break
            
        # Generate responses
        query_tensors = tokenizer(
            batch["query"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate model responses
        response_tensors = trainer.generate(
            query_tensors=query_tensors,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        
        # Decode responses
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute rewards
        rewards = combine_rewards(
            responses=responses, 
            query=batch["query"], 
            answer=batch["answer"]
        )
        
        # Train model with PPO
        train_stats = trainer.step(query_tensors, response_tensors, rewards)
        
        # Log training statistics
        trainer.log_stats(
            stats=train_stats,
            batch={"query": batch["query"], "response": responses},
            rewards=rewards,
        )
        
        # Save model checkpoint
        if step % ppo_config.save_steps == 0:
            trainer.save_pretrained(f"{ppo_config.output_dir}/checkpoint-{step}")

    # Save final model
    trainer.save_pretrained(f"{ppo_config.output_dir}/final")


if __name__ == "__main__":
    parser = TrlParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    main(script_args, training_args, model_args) 