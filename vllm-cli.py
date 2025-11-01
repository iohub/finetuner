import os
import argparse
from typing import List, Dict, Any
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
SYSTEM_PROMPT_FILE = "system.prompt"

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
    return OpenAI(
        base_url=f"{host}/v1",
        api_key="EMPTY" 
    )

def load_system_prompt() -> str | None:
    """尝试从当前目录加载 system.prompt 文件内容"""
    if os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content
                else:
                    console.print(f"[bold yellow]警告:[/bold yellow] {SYSTEM_PROMPT_FILE} 文件为空。")
                    return None
        except Exception as e:
            console.print(f"[bold red]加载 {SYSTEM_PROMPT_FILE} 失败:[/bold red] {e}")
            return None
    return None


# --- 核心对话逻辑 ---

def chat_loop(client: OpenAI, model: str, max_tokens: int):
    """
    多轮对话循环。
    """
    
    history: List[Dict[str, str]] = []
    
    # 1. 加载系统提示
    system_prompt = load_system_prompt()
    system_prompt_status = "[dim]未加载 system.prompt 文件[/dim]"
    
    if system_prompt:
        # 如果加载了系统提示，将其作为第一个消息添加到历史记录中
        history.append({"role": "system", "content": system_prompt})
        system_prompt_status = f"[bold green]已加载 system.prompt[/bold green] ({len(system_prompt)} 字符)"
        console.print(Panel(f"[bold yellow]System Prompt 内容摘要:[/bold yellow]\n{system_prompt[:100]}...", border_style="yellow"))


    # 2. 打印欢迎信息
    console.print(
        Panel(
            f"[bold cyan]VLLM Chat 客户端[/bold cyan]\n"
            f"[yellow]模型:[/yellow] {model}\n"
            f"[yellow]最大 tokens:[/yellow] {max_tokens}\n"
            f"[yellow]System Prompt:[/yellow] {system_prompt_status}\n"
            f"\n[dim]输入 'exit' 或 'quit' 退出, 'clear' 清空历史。[/dim]",
            title="✨ 欢迎",
            border_style="cyan"
        )
    )

    while True:
        # 3. 获取用户输入
        try:
            user_input = console.input("[bold green]您 (User):[/bold green] ")
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if user_input.lower() in ["quit", "exit"]:
            break
        
        if user_input.lower() == "clear":
            # 清空历史，但保留 system prompt (如果存在)
            history = history[:1] if history and history[0]["role"] == "system" else []
            console.print(Panel("✅ [bold yellow]对话历史已清空。[/bold yellow]", border_style="yellow"))
            continue

        if not user_input.strip():
            continue

        # 4. 更新对话历史 (用户输入)
        history.append({"role": "user", "content": user_input})

        # 5. 发起流式 API 请求
        try:
            console.print("\n[bold magenta]AI (Assistant):[/bold magenta] ", end="")
            
            # 使用 `stream=True` 开启流式传输
            stream = client.chat.completions.create(
                model=model,
                messages=history, # 包含 system prompt 和所有历史记录
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
            )
            
            full_response = ""
            
            # 6. 打印流式回复
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    console.print(content, end="", soft_wrap=True)
                    full_response += content

            console.print() 
            
            # 7. 将完整的助手回复添加到历史记录中
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
        # 1. 初始化客户端
        openai_client = init_client(args.host)
        
        # 2. 启动对话循环
        chat_loop(openai_client, args.model, args.max_tokens)
        
    except Exception as e:
        console.print(Panel(f"[bold red]客户端启动失败:[/bold red] {e}", border_style="red"))