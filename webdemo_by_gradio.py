import gradio as gr
from component.llms import Openai_model
import time

custom_url = 'https://aihubmix.com/v1'
model=Openai_model(base_url=custom_url)

def echo(message, history):
    # 修改Openai_model类的chat方法，使其返回info和result
    info = model.db.query(message, model.embedding_model, 1)
    result = model.chat(message)
    # 将info作为参考信息展示
    return f"**参考信息：**\n```\n{info}\n```\n\n**回答：**\n{result}"


demo = gr.ChatInterface(fn=echo, 
                        examples=["中华人民共和国消费者权益保护法什么时候,在哪个会议上通过的？", "中华人民共和国消费者权益保护的目录是什么？","RinyRAG的项目结构是怎么样的"], 
                        title="Echo Bot",
                        theme="soft")
demo.launch()