

### **gsm8k效果评估**

| 模型版本 | 数据量 (num) | 正确解答数量 | 准确率 (Accuracy) | 提升值 |
| :--- | :--- | :--- | :--- | :--- |
| **eval-Qwen2.5-3B-Instruct-epoch0.7** | 100 | 74 | $0.7327$ | +0.08  |
| **eval-Qwen2.5-3B-Instruct** | 100 | 66 | $0.6535$ | |
| **eval-Qwen2.5-3B-Instruct-epoch0.7** | 200 | 142 | $0.7065$ | +0.06 |
| **eval-Qwen2.5-3B-Instruct** | 200 | 130 | $0.6468$ | |
| **eval-Qwen2.5-3B-Instruct-epoch0.7** | 500 | 346 | $0.69$ | +0.07 |
| **eval-Qwen2.5-3B-Instruct** | 500 | 311 | $0.62$ | |
---

### **总结**

* **模型差异：** 经过 $0.7$ 个 epoch 训练的模型（**eval-Qwen2.5-3B-Instruct-epoch0.7**）表现**优于**基础模型（**eval-Qwen2.5-3B-Instruct**）。
