"""
Fine-tuning Pipeline for LLMs

Implements LoRA/QLoRA fine-tuning for clinical trial data extraction.

References:
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    # Model configuration
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_cache_dir: Optional[str] = None

    # LoRA configuration
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha parameter
    lora_dropout: float = 0.05  # Dropout rate
    lora_target_modules: List[str] = None  # Target modules for LoRA

    # Training configuration
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Quantization (for QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Additional options
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4


class DataPreparation:
    """
    Prepare data for fine-tuning.

    Converts annotated data to training format.
    """

    @staticmethod
    def load_annotated_dataset(
        file_path: str
    ) -> List[Dict]:
        """
        Load annotated dataset from JSON file.

        Args:
            file_path: Path to annotated dataset

        Returns:
            List of annotated examples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Handle different JSON structures
            if 'examples' in data:
                return data['examples']
            elif 'data' in data:
                return data['data']
            else:
                return list(data.values())

        return data

    @staticmethod
    def format_for_training(
        examples: List[Dict],
        template_type: str = "alpaca"
    ) -> List[Dict]:
        """
        Format examples for training.

        Args:
            examples: List of annotated examples
            template_type: Type of prompt template ('alpaca', 'chatml', 'custom')

        Returns:
            List of formatted training examples
        """
        formatted = []

        for example in examples:
            if template_type == "alpaca":
                formatted.append(DataPreparation._format_alpaca(example))
            elif template_type == "chatml":
                formatted.append(DataPreparation._format_chatml(example))
            else:
                formatted.append(DataPreparation._format_custom(example))

        return formatted

    @staticmethod
    def _format_alpaca(example: Dict) -> Dict:
        """Format in Alpaca style."""
        return {
            'instruction': example.get('instruction', example.get('task', '')),
            'input': example.get('input', example.get('study_text', '')),
            'output': example.get('output', example.get('extraction', ''))
        }

    @staticmethod
    def _format_chatml(example: Dict) -> Dict:
        """Format in ChatML style."""
        messages = []

        # System message
        messages.append({
            'role': 'system',
            'content': 'You are a clinical research data extraction expert.'
        })

        # User message
        user_content = f"""Task: {example.get('task', '')}
Study: {example.get('study_title', '')}
Outcome: {example.get('outcome', '')}
Intervention: {example.get('intervention', '')}
Comparator: {example.get('comparator', '')}

Study Text:
{example.get('study_text', '')}"""

        messages.append({
            'role': 'user',
            'content': user_content
        })

        # Assistant message
        messages.append({
            'role': 'assistant',
            'content': example.get('output', example.get('extraction', ''))
        })

        return {'messages': messages}

    @staticmethod
    def _format_custom(example: Dict) -> Dict:
        """Custom formatting for specific use case."""
        # Concatenate all fields into a single text
        text_parts = []

        if example.get('instruction'):
            text_parts.append(f"Instruction: {example['instruction']}")

        if example.get('study_text'):
            text_parts.append(f"Text: {example['study_text']}")

        text_parts.append(f"Output: {example.get('output', example.get('extraction', ''))}")

        return {'text': '\n\n'.join(text_parts)}

    @staticmethod
    def create_train_val_split(
        examples: List[Dict],
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train/validation split.

        Args:
            examples: List of examples
            train_ratio: Proportion for training
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_examples, val_examples)
        """
        np.random.seed(random_seed)
        indices = np.random.permutation(len(examples))

        n_train = int(len(examples) * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_examples = [examples[i] for i in train_indices]
        val_examples = [examples[i] for i in val_indices]

        return train_examples, val_examples

    @staticmethod
    def create_dataset(
        examples: List[Dict],
        tokenizer,
        max_length: int = 2048
    ) -> Dataset:
        """
        Create HuggingFace dataset from examples.

        Args:
            examples: List of formatted examples
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            HuggingFace Dataset
        """
        def tokenize_function(examples_dict):
            # Determine the format
            if 'text' in examples_dict:
                # Simple text format
                return tokenizer(
                    examples_dict['text'],
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )
            elif 'messages' in examples_dict:
                # Chat format
                # Apply chat template
                texts = [
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    for messages in examples_dict['messages']
                ]
                return tokenizer(
                    texts,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )
            else:
                # Alpaca format
                instructions = examples_dict['instruction']
                inputs = examples_dict['input']
                outputs = examples_dict['output']

                prompts = []
                for inst, inp, out in zip(instructions, inputs, outputs):
                    if inp:
                        prompt = f"""### Instruction:
{inst}

### Input:
{inp}

### Response:
{out}"""
                    else:
                        prompt = f"""### Instruction:
{inst}

### Response:
{out}"""
                    prompts.append(prompt)

                return tokenizer(
                    prompts,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )

        # Create dataset
        dataset_dict = {k: [d[k] for d in examples] for k in examples[0].keys()}
        dataset = Dataset.from_dict(dataset_dict)

        # Tokenize
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        return tokenized


class FineTuner:
    """
    Fine-tune LLMs using LoRA/QLoRA.
    """

    def __init__(self, config: FineTuningConfig):
        """
        Initialize fine-tuner.

        Args:
            config: FineTuning configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Install with: pip install transformers torch"
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self) -> None:
        """Load base model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.config.use_4bit and BITSANDBYTES_AVAILABLE and BitsAndBytesConfig is not None:
            # QLoRA: 4-bit quantization
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.model_cache_dir,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
                trust_remote_code=True
            )

            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)

        else:
            # Standard loading
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.model_cache_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        self.model = model

    def setup_lora(self) -> None:
        """Set up LoRA adapters."""
        if self.config.lora_target_modules is None:
            # Default target modules for Mistral/Llama
            self.config.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training metrics
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="none",  # Set to "wandb" for Weights & Biases logging
            fp16=False,
            bf16=True,
            optim="paged_adamw_32bit",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # Train
        result = self.trainer.train()

        return {
            'train_loss': result.training_loss,
            'train_samples': len(train_dataset),
            'epochs': self.config.num_train_epochs
        }

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save fine-tuned model.

        Args:
            output_dir: Directory to save model (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_finetuned_model(self, checkpoint_path: str) -> None:
        """
        Load fine-tuned model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )


def fine_tune_model(
    annotated_data_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "./checkpoints",
    use_4bit: bool = True
) -> str:
    """
    Fine-tune a model on annotated clinical trial data.

    Args:
        annotated_data_path: Path to annotated JSON dataset
        base_model: Base model to fine-tune
        output_dir: Directory to save checkpoints
        use_4bit: Whether to use QLoRA (4-bit quantization)

    Returns:
        Path to fine-tuned model checkpoint
    """
    # Load configuration
    config = FineTuningConfig(
        base_model=base_model,
        output_dir=output_dir,
        use_4bit=use_4bit
    )

    # Initialize fine-tuner
    tuner = FineTuner(config)

    # Load model
    tuner.load_model()

    # Setup LoRA
    tuner.setup_lora()

    # Load and prepare data
    examples = DataPreparation.load_annotated_dataset(annotated_data_path)
    formatted = DataPreparation.format_for_training(examples, template_type="alpaca")
    train_examples, val_examples = DataPreparation.create_train_val_split(formatted)

    # Create datasets
    train_dataset = DataPreparation.create_dataset(
        train_examples,
        tuner.tokenizer,
        config.max_seq_length
    )
    val_dataset = DataPreparation.create_dataset(
        val_examples,
        tuner.tokenizer,
        config.max_seq_length
    )

    # Train
    metrics = tuner.train(train_dataset, val_dataset)

    # Save model
    tuner.save_model()

    return output_dir


if __name__ == "__main__":
    print("Fine-tuning Pipeline Module loaded")
    print("Features:")
    print("  - LoRA/QLoRA fine-tuning")
    print("  - Multiple prompt templates")
    print("  - Train/validation splitting")
    print("  - HuggingFace Trainer integration")
