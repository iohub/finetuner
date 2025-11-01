
from unsloth import FastLanguageModel
import re, sys, os
from datasets import load_dataset, Dataset, concatenate_datasets

# TORCH_COMPILE_DISABLE=1
# TORCH_INDUCTOR_DISABLE_CUDAGRAPHS=1

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

def load_model():
    model_path = '/home/do/ssd/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct'

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = False, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        # enforce_eager=True,
        # gpu_memory_utilization = 0.85, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    return model, tokenizer


# data prepare


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_hash_answer_zh(text: str) -> str | None:
    if not text or '答案是：' not in text:
        return None
    return text.split('答案是：')[1].strip()


def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('meta-math/GSM8K_zh', 'default')[split] # type: ignore
    
    # 1. 1/4 Chinese Corpus 和 3/4 English Corpus
    data_splits = data.train_test_split(test_size=0.75, seed=42)
    print(data_splits)

    data_zh = data_splits['test'].map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh']}
        ],
        'answer': extract_hash_answer_zh(x['answer_zh'])
    }, num_proc=4) # type: ignore (可以添加 num_proc 加快处理)
    data_zh = data_zh.filter(lambda x: x['answer'] is not None, num_proc=4)
    print(data_zh[:10])
    
    data_en = data_splits['train'].map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }, num_proc=4) # type: ignore (可以添加 num_proc 加快处理)
    data_en = data_en.filter(lambda x: x['answer'] is not None, num_proc=4)
    print('='*50)
    print(data_en[:10])

    combined_data = concatenate_datasets([data_zh, data_en])
    combined_data = combined_data.shuffle(seed=42)
    return combined_data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def count_chinese_by_regex(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    return len(chinese_chars)


def reward_lang(text) -> float:
    score = 0.125
    chinese_count = count_chinese_by_regex(text)
    if chinese_count > len(text) * 0.85:
        score = 3
    elif chinese_count > len(text) * 0.5:
        score = 2
    elif chinese_count > len(text) * 0.2:
        score = 1
    else:
        score = 0.5

    return score

def lang_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [reward_lang(c) for c in contents]


# tranning model

max_prompt_length = 512
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1200,
    save_steps = 400,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

model, tokenizer = load_model()

print('train datasets:{}'.format(len(dataset)))

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        lang_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train(resume_from_checkpoint="outputs/checkpoint-800")

# save model

model.save_pretrained_merged("ft_model", tokenizer, save_method = "merged_16bit",)


