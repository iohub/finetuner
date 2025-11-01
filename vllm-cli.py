import os
import argparse
from typing import List, Dict, Any

# 使用 openai 库与 vllm 的 OpenAI 兼容 API 进行交互
from openai import OpenAI

# 使用 rich 库美化 CLI 界面
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# 初始化 rich console
console = Console()

# --- 配置参数与初始化 ---

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="🚀 VLLM 多轮对话客户端 (兼容 OpenAI API)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="VLLM 服务的地址和端口 (默认: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default_vllm_model",  # 根据您的 vLLM 服务配置的模型名称修改
        help="要使用的模型名称 (默认: default_vllm_model)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="生成回复的最大 token 数 (默认: 512)"
    )
    return parser.parse_args()

def init_client(host: str) -> OpenAI:
    """初始化 OpenAI 客户端，指向 VLLM 服务"""
    # VLLM API 兼容 OpenAI，通常不需要 API Key，传入空字符串即可
    return OpenAI(
        base_url=f"{host}/v1",
        api_key="EMPTY" 
    )

# --- 核心对话逻辑 ---

def chat_loop(client: OpenAI, model: str, max_tokens: int):
    """
    多轮对话循环。
    
    :param client: OpenAI 客户端实例
    :param model: 模型名称
    :param max_tokens: 最大生成 token 数
    """
    
    # 存储对话历史，格式兼容 OpenAI Chat Completions API
    # [{"role": "user", "content": "Hello"}, ...]
    history: List[Dict[str, str]] = []
    
    console.print(
        Panel(
            f"[bold cyan]VLLM Chat 客户端[/bold cyan]\n"
            f"[yellow]模型:[/yellow] {model}\n"
            f"[yellow]最大 tokens:[/yellow] {max_tokens}\n"
            f"\n[dim]输入 'exit' 或 'quit' 退出, 'clear' 清空历史。[/dim]",
            title="✨ 欢迎",
            border_style="cyan"
        )
    )

    while True:
        # 1. 获取用户输入 (使用 input() 获取用户输入)
        try:
            user_input = console.input("[bold green]您 (User):[/bold green] ")
        except EOFError:
            # 捕获 CTRL+D
            break
        except KeyboardInterrupt:
            # 捕获 CTRL+C
            break

        if user_input.lower() in ["quit", "exit"]:
            break
        
        if user_input.lower() == "clear":
            history = []
            console.print(Panel("✅ [bold yellow]对话历史已清空。[/bold yellow]", border_style="yellow"))
            continue

        if not user_input.strip():
            continue

        # 2. 更新对话历史
        history.append({"role": "user", "content": user_input})

        # 3. 发起流式 API 请求
        try:
            console.print("\n[bold magenta]AI (Assistant):[/bold magenta] ", end="")
            
            # 使用 `stream=True` 开启流式传输
            stream = client.chat.completions.create(
                model=model,
                messages=history,
                max_tokens=max_tokens,
                temperature=0.0, # 默认低温度，方便测试
                stream=True,     # 关键：启用流式输出
            )
            
            # 用于存储完整的助手回复
            full_response = ""
            
            # 4. 打印流式回复
            for chunk in stream:
                # 从流中获取内容
                content = chunk.choices[0].delta.content
                if content:
                    # 使用 rich.print() 的实时打印功能
                    console.print(content, end="", soft_wrap=True)
                    full_response += content

            console.print() 
            
            # 5. 将完整的助手回复添加到历史记录中
            history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            # 打印错误信息并移除最后一条用户输入，避免下次循环重复发送
            console.print(Panel(f"[bold red]API 请求出错:[/bold red] {e}", border_style="red"))
            if history and history[-1]["role"] == "user":
                history.pop()

    console.print(Panel("👋 [bold cyan]感谢使用，再见！[/bold cyan]", border_style="cyan"))


if __name__ == "__main__":
    args = parse_args()
    
    try:
        openai_client = init_client(args.host)
        chat_loop(openai_client, args.model, args.max_tokens)
        
    except Exception as e:
        console.print(Panel(f"[bold red]客户端启动失败:[/bold red] {e}", border_style="red"))