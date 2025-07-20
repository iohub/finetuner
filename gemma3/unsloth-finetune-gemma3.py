from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import os
import time
import glob
from transformers import TrainerCallback

# 自定义回调函数，在每次保存后让GPU休息15秒
class GPURestCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print("模型已保存，GPU休息15秒...")
        time.sleep(15)
        print("GPU休息完成，继续训练")

# 检查和获取最新的checkpoint
def get_latest_checkpoint(output_dir):
    """
    获取输出目录中最新的checkpoint路径
    """
    if not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在，将从头开始训练")
        return None
    
    # 查找所有checkpoint目录
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"在 {output_dir} 中未找到checkpoint，将从头开始训练")
        return None
    
    # 按checkpoint编号排序，获取最新的
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]
    
    print(f"找到最新checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# Fix for dynamo recompilation limit issue
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256
# Alternative: Disable torch compile entirely (uncomment if cache increase doesn't work)
# torch._dynamo.config.disable = True


model, tokenizer = FastModel.from_pretrained(
    model_name="/media/do/llmhub/modelhub/gemma-3-4b-it",
    dtype = None,
    max_seq_length=2048,  # 从5000降低到2048，平衡性能和内存
    load_in_4bit=False,
    load_in_8bit=True,
    full_finetuning=False
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True, 

    r = 16,           # 增加r值以提高模型容量
    lora_alpha = 32,  # alpha = 2 * r 通常效果更好
    lora_dropout = 0,  # 改回0以获得最佳Unsloth性能
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

# 添加训练/验证数据分割
train_dataset = dataset.select(range(int(len(dataset) * 0.95)))
eval_dataset = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))

# 定义输出目录
output_dir = "./training_outputs"

# 检查是否有可以恢复的checkpoint
latest_checkpoint = get_latest_checkpoint(output_dir)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,  # 使用分割后的训练数据
    eval_dataset = eval_dataset,    # 添加评估数据集
    args = SFTConfig(
        dataset_text_field = "text",
        output_dir = output_dir,            # 输出目录，TensorBoard日志将保存在这里
        logging_dir = "./training_logs",    # TensorBoard日志目录
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 50,              # 大幅增加warmup步数
        num_train_epochs = 1.0,         # 先训练1个epoch观察效果
        # max_steps = 30,
        learning_rate = 5e-5,           # 大幅降低学习率
        logging_steps = 1,
        logging_strategy = "steps",         # 按步数记录日志
        eval_steps = 50,                    # 每50步评估一次
        save_steps = 100,                   # 增加保存间隔
        save_strategy = "steps",            # 按步数保存
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",       # 改用cosine调度器
        seed = 3407,
        report_to = "tensorboard",          # 使用TensorBoard进行监控
        run_name = "gemma3-finetune-v2",
        dataloader_num_workers = 2,         # 数据加载器的工作进程数
        # fp16 = True,                      # Gemma3不支持fp16，自动使用float32
        gradient_checkpointing = True,      # 启用梯度检查点节省内存
    ),
    callbacks = [GPURestCallback()], # 添加自定义回调
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

# 开始或恢复训练
if latest_checkpoint:
    print(f"从checkpoint恢复训练: {latest_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("开始新的训练")
    trainer_stats = trainer.train()





