# 加载环境变量
from chat.chat import Chat
import streamlit as st
import os
import openai
from dotenv import load_dotenv, find_dotenv

from utils.print_utils import color_print
_ = load_dotenv(find_dotenv())

from agent.auto_gpt import AutoGPT
from langchain_openai import ChatOpenAI
from tools import *
from tools.python_tool import ExcelAnalyser
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


def launch_agent(agent: AutoGPT,chat: Chat):
    st.title('🦜🔗 My-Ai-Agent')
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["chat", "agent"],
        captions = ["问答机器人", "Agent对话模式"])
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    chat_history = ChatMessageHistory()
    messages = st.container(height=300)
    if task := st.chat_input("请输入你的问题~"):
        color_print(text=f"{human_icon}：{task}\n",)
        st.session_state.messages.append({"role": "user", "text": task})
        if task.strip().lower() == "quit":
            return
        if selected_method=="chat":
            reply = chat.run(task)
        elif selected_method=="agent":
            # 运行智能体
            reply = agent.run(task, chat_history, verbose=True)
        st.session_state.messages.append({"role": "assistant", "text": reply})

        # 写入文件
        color_print(text=f"{ai_icon}：{reply}\n\n")
        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])  


def main():
    # 设置代理
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url is not None:
        openai.api_base = proxy_url
        
    # 语言模型
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        model_kwargs={
            "seed": 42
        },
    )

    # 自定义工具集
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            prompt_file="./prompts/tools/excel_analyser.txt",
            verbose=True
        ).as_tool()
    ]

    # 定义智能体
    agent = AutoGPT(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        max_thought_steps=20,
    )
    chat = Chat(
        llm=llm,
    )
    # 运行智能体
    launch_agent(agent,chat)


if __name__ == "__main__":
    main()
