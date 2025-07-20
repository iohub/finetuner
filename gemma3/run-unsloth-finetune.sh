

# speed up huggingface downloads
export HF_ENDPOINT=https://hf-mirror.com

# use bigger VRAM device
export CUDA_VISIBLE_DEVICES=1

# run the finetuning
python unsloth-finetune-gemma3.py

