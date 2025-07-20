from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import os

# Fix for dynamo recompilation limit issue
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256
# Alternative: Disable torch compile entirely (uncomment if cache increase doesn't work)
# torch._dynamo.config.disable = True


model, tokenizer = FastModel.from_pretrained(
    model_name="/media/do/llmhub/modelhub/gemma-3-4b-it",
    dtype = None,
    max_seq_length=1024,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True, 

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

model.print_trainable_parameters()


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = standardize_data_formats(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        output_dir = "./training_outputs",  # 输出目录，TensorBoard日志将保存在这里
        logging_dir = "./training_logs",    # TensorBoard日志目录
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        logging_strategy = "steps",         # 按步数记录日志
        save_steps = 10,                    # 每10步保存一次模型
        save_strategy = "steps",            # 按步数保存
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "tensorboard",          # 使用TensorBoard进行监控
        run_name = "gemma3-finetune",       # 运行名称
        dataloader_num_workers = 2,         # 数据加载器的工作进程数
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()





