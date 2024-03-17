# Training PEFT models with new tokens being added to the embedding layers and tokenizer
# In this example, we will learn how to train a LoRA model when adding new tokens to the tokenizer and model. This is a common usecase when doing the following:

# 1. Instruction finetuning with new tokens beind added such as <|user|>, <|assistant|>, <|system|>, </s>, <s> to properly format the conversations
# 2. Finetuning on a specific language wherein language spoecific tokens are added, e.g., korean tokens being added to vocabulary for finetuning LLM on Korean datasets.
# 3. Instruction finetuning to return outputs in certain format to enable agent behaviour new tokens such as <|FUNCTIONS|>, <|BROWSE|>, <|TEXT2IMAGE|>, <|ASR|>, <|TTS|>, <|GENERATECODE|>, <|RAG|>.
# In such cases, you add the Embedding modules to the LORA target_modules. PEFT will take care of saving the embedding layers with the new added tokens along with the adapter weights 
# that were trained on the specific initialization of the embeddings weights of the added tokens.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PeftExamples"
import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import torch
from dataclasses import dataclass, field
from typing import Optional
from dataclass_csv import DataclassReader
from torch.utils.data import Dataset, DataLoader

from enum import Enum


### Prepare Model and Tokenizer ###

class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"
    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"
    begin_dsts = "<|begindsts|>"
    end_dsts = "<|enddsts|>"
    begin_dst = "<|begindst|>"
    end_dst = "<|enddst|>"
    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"
    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"
    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"
    begin_user_action = "<|beginuseraction|>"
    end_user_action = "<|enduseraction|>"
    sys_actions = "<|sysactions|>"
    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"
    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"
    pad_token = "<|pad|>"
    bos_token = "<|startoftext|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    pad_token=SpecialTokens.pad_token.value,
    bos_token=SpecialTokens.bos_token.value,
    eos_token=SpecialTokens.end_target.value,
    additional_special_tokens=SpecialTokens.list(),
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
    # use_flash_attention_2=True, # leading to an error
)
model.resize_token_embeddings(len(tokenizer))


model_mistral = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
    # use_flash_attention_2=True, # leading to an error
)
model_mistral.resize_token_embeddings(len(tokenizer))


### Apply LoRA ###

config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
)
model = get_peft_model(model, config)
print(model.print_trainable_parameters())
print(model)
from fvcore.nn.parameter_count import parameter_count_table
print(parameter_count_table(model, max_depth=4))

### Prepare Dataset ###
from datasets import load_dataset

dataset = load_dataset("smangrul/assistant_chatbot_dataset")
dataset = dataset["train"].train_test_split(0.2)

text_column = "context"
label_column = "target"
max_length = 512


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(examples[text_column])
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=8, pin_memory=True
)

next(iter(train_dataloader))

tokenizer.decode(train_dataset[0]["input_ids"])


### Train the model ###

training_args = TrainingArguments(
    output_dir="mistral_lora_clm_with_added_tokens",
    num_train_epochs=2,
    save_total_limit=5,
    per_device_train_batch_size=8,
    warmup_steps=10,
    weight_decay=0.0001,
    dataloader_drop_last=True,
    bf16=True,
    logging_steps=10,
    learning_rate=1e-5,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    hub_model_id="smangrul/mistral_lora_clm_with_added_tokens",
    push_to_hub=True,
    hub_private_repo=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)
# model.config.use_cache = False
trainer.train()



### Check the model output on a sample from evaluation dataset ###
import random

# i = random.randint(0, len(dataset["test"]))
i = 0
context = dataset["test"][i]["context"]

batch = tokenizer(context, return_tensors="pt")
batch = {k: v.to("cuda") for k, v in batch.items()}
model.eval()
model.to("cuda")
output_tokens = model.generate(
    **batch,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
target_predicted = tokenizer.decode(output_tokens[0], skip_special_tokens=False).split("<|endcontext|>")[1]
target = dataset["test"][i]["target"]
print(f"{context=} \n\n {target_predicted=} \n\n {target=}")


model_mistral.eval()
model_mistral.to("cuda")
output_tokens = model_mistral.generate(
    **batch,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

target_predicted_mistral = tokenizer.decode(output_tokens[0], skip_special_tokens=False).split("<|endcontext|>")[1]
print(target_predicted_mistral)
