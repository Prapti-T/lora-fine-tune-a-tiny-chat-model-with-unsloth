"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        # If bitsandbytes is not installed, 4-bit layers cannot exist
        return False

    # Walk through all submodules in the model hierarchy
    for module in model.modules():
        if isinstance(module, Linear4bit):
            return True
            
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, 'eos_token_id'):
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",]
                #   "gate_proj", "up_proj", "down_proj",],
    return target_modules

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()
    return FastLanguageModel.get_peft_model(
        model, r, target_modules, lora_alpha
    )

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return float(trainable_count / total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "Summarize: Water freezes at 0 degrees Celsius.",
            "response": "Water turns into ice at 0 degrees Celsius.",
        },
        {
            "instruction": "What is 7 plus 5?",
            "response": "7 plus 5 is 12.",
        },
        {
            "instruction": "Name one primary color.",
            "response": "Blue is a primary color.",
        },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    ins = example["instruction"]
    res = example["response"]
    return f"### Instruction:\n{ins}\n\n### Response:\n{res}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    op = [format_instruction_example(ex) for ex in examples]
    return op

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    return Dataset.from_dict({'text':texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)['input_ids']

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
import torch
from transformers import TrainingArguments

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # TODO: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    query = torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=query,
        fp16=not query,
        logging_steps=1,
        optim="adamw_8bit",
    )

# Step 16 - build_sft_trainer
from trl import SFTTrainer

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    # TODO: wire model, tokenizer, dataset, and training_args into an SFTTrainer
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False
    )

# Step 17 - run_sft_training
import sys

def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    # TODO: drive the trainer through its short optimization run and return the final loss
    args = getattr(trainer, "args", None)

    if args is not None:
        args_class = args.__class__
        module = sys.modules.get(args_class.__module__)

        if module is not None:
            setattr(module, args_class.__name__, args_class)

    train_result = trainer.train()

    return float(train_result.training_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    # TODO: call the Unsloth helper that prepares the model for fast generation
    model = FastLanguageModel.for_inference(model)
    return model

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    # TODO: wrap the instruction as a user turn and produce the assistant-generation prompt string
    messages = [{"role": "user", "content": instruction}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# Step 20 - generate_reply
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    # TODO: tokenize prompt, run model.generate with do_sample=False, decode new tokens only
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True)

