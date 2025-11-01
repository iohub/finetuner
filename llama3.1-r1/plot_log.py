import json
import pandas as pd
import os, sys
from tensorboardX import SummaryWriter
from datetime import datetime

# ----------------- 配置 -----------------
# 你的 trainer_state.json 文件路径
JSON_FILE_PATH = sys.argv[1]
# TensorBoard 日志文件的保存目录
LOG_DIR = "runs/log_history_plot_" + datetime.now().strftime("%Y%m%d-%H%M%S")
# ----------------------------------------
# tensorboard --host 0.0.0.0 --logdir runs

def write_log_history_to_tensorboard(json_file_path, log_dir):
    """
    读取 trainer_state.json 中的 log_history，并将其写入 TensorBoard 事件文件。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 {json_file_path}。请检查路径。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 格式不正确。")
        return

    log_history = data.get('log_history', [])
    if not log_history:
        print("警告：log_history 列表为空，没有数据可供绘制。")
        return

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir)
    print(f"正在将日志写入 TensorBoard 目录: {log_dir}")

    for log_entry in log_history:
        # 使用 'step' 作为 TensorBoard 的全局步数 (global_step)
        step = log_entry.get('step')
        if step is None:
            # 如果没有 step，跳过此条日志或使用 epoch * 步数进行估算
            continue
        
        # 遍历日志条目中的所有键值对
        for key, value in log_entry.items():
            # 过滤掉非数值类型和不需要的键
            if isinstance(value, (int, float)) and key not in ['step', 'epoch', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second']:
                # TensorBoard 通常使用斜杠来分组。
                # 训练集指标通常有 'loss'，验证集指标通常有 'eval_loss' 等。
                if key.startswith('eval_'):
                    tag = f"eval/{key.replace('eval_', '')}"
                elif key == 'loss' or key == 'learning_rate':
                    tag = f"train/{key}"
                elif key.startswith('rewards'):
                    tag = f"rewards/{key}"
                else:
                    tag = f"misc/{key}"
                    
                writer.add_scalar(tag, value, step)

    writer.close()
    print("日志写入完成。")

if __name__ == '__main__':
    write_log_history_to_tensorboard(JSON_FILE_PATH, LOG_DIR)