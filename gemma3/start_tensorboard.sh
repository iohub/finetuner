#!/bin/bash

# 确保TensorBoard已安装
echo "检查TensorBoard是否已安装..."
if ! command -v tensorboard &> /dev/null; then
    echo "TensorBoard未安装，正在安装..."
    pip install tensorboard
else
    echo "TensorBoard已安装"
fi

# 创建日志目录（如果不存在）
mkdir -p ./training_logs
mkdir -p ./training_outputs

echo "启动TensorBoard..."
echo "请在浏览器中打开: http://localhost:6006"
echo "按 Ctrl+C 停止TensorBoard"
echo ""

# 启动TensorBoard
tensorboard --logdir=./training_logs --port=6006 --host=0.0.0.0 