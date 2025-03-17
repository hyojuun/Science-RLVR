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

import datasets
import torch
import transformers
import pandas as pd
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import PPOConfig
from open_r1.rewards import (
    accuracy_reward,
    format_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, ModelConfig, ScriptArguments, TrlParser, get_peft_config

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


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    train_dataset = pd.read_csv("/data/train.csv")
    valid_dataset = pd.read_csv("/data/validation.csv")

    # Convert pandas dataframes to huggingface datasets
    train_dataset = datasets.Dataset.from_pandas(train_dataset)
    valid_dataset = datasets.Dataset.from_pandas(valid_dataset)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format data into the required format for PPO training
    def format_data(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    train_dataset = train_dataset.map(format_data)
    valid_dataset = valid_dataset.map(format_data)

    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("messages")
    if "messages" in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the PPO trainer
    #############################
    # Initialize the model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Initialize PPO trainer
    ppo_config = PPOConfig(
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.mini_batch_size,
        batch_size=training_args.batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=training_args.ppo_epochs,
        max_steps=training_args.max_steps,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=None,
        peft_config=get_peft_config(model_args) if hasattr(model_args, "use_peft") and model_args.use_peft else None,
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(trainer.dataloader):
            if step >= training_args.max_steps:
                break

            # Generate responses
            query_tensors = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=training_args.max_prompt_length
            ).to(model.device)

            # Generate model responses
            response_tensors = trainer.generate(
                query_tensors=query_tensors,
                max_new_tokens=training_args.max_completion_length,
                do_sample=True,
                temperature=0.7,
            )

            # Decode responses
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute rewards using multiple reward functions
            rewards = []
            for reward_func in reward_funcs:
                batch_rewards = reward_func(responses, **batch)
                rewards.append(batch_rewards)

            # Average the rewards
            final_rewards = torch.zeros(len(responses))
            for i in range(len(responses)):
                reward_sum = 0
                for j in range(len(rewards)):
                    reward_sum += rewards[j][i]
                final_rewards[i] = reward_sum / len(rewards)

            # Train model with PPO
            train_stats = trainer.step(query_tensors, response_tensors, final_rewards)

            # Log training statistics
            trainer.log_stats(
                stats=train_stats,
                batch={"query": batch["prompt"], "response": responses},
                rewards=final_rewards,
            )

            # Save model checkpoint
            if step % training_args.save_steps == 0:
                trainer.save_pretrained(os.path.join(training_args.output_dir, f"checkpoint-{step}"))

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_path": "/data/train.csv",
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        model.config.use_cache = True
        model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)