from openai import OpenAI
from langchain.prompts import  PromptTemplate
from component.embedding import Zhipuembedding,Jinaembedding
from component.databases import Vectordatabase,ImageVectordatabase
import os
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

class Openai_model:
    def __init__(self,model_name:str='gpt-4o-mini',temperature:float=0.9, base_url:str=None) -> None:
        
        #初始化大模型
        self.model_name = model_name  # Store model_name
        self.temperature = temperature  # Store temperature
        if base_url:
            self.client = OpenAI(base_url=base_url) # Initialize official OpenAI client
        else:
            self.client = OpenAI() # Initialize official OpenAI client

        #加载向量数据库，embedding模型
        self.db=Vectordatabase()
        self.db.load_vector()
        self.embedding_model=Jinaembedding(api_key=OpenAI.api_key)
        
    #定义chat方法
    def chat(self,question:str):
        #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""

        info=self.db.query(question, self.embedding_model, 1)

        prompt=PromptTemplate(template=template, input_variables=["question","info"]).format(question=question, info=info)

        # Call the official OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        res = completion.choices[0].message.content

        return  res


class OpenAI_model_vision:
    def __init__(self,model_name:str='gpt-4o-mini',temperature:float=0.9, base_url:str=None) -> None:

        #初始化大模型
        self.model_name = model_name  # Store model_name
        self.temperature = temperature  # Store temperature
        if base_url:
            self.client = OpenAI(base_url=base_url) # Initialize official OpenAI client
        else:
            self.client = OpenAI() # Initialize official OpenAI client
        #加载向量数据库，embedding模型
        self.db=ImageVectordatabase()
        self.db.load_vector()
        api_key = OpenAI.api_key
        self.embedding_model=Jinaembedding(api_key=api_key)

    def chat_vision(self,question:str):
        #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
        info=self.db.query(question, self.embedding_model, 3)
        
        prompt=PromptTemplate(template=template, input_variables=["question","info"]).format(question=question, info=info)
        
        # 准备消息列表，首先添加系统提示
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 如果info中包含图片名称，从images文件夹加载图片
        image_names = [item[0] for item in info] if isinstance(info, list) and len(info) > 0 and isinstance(info[0], tuple) else []
        
        # 构建多模态消息
        content = []
        content.append({"type": "text", "text": prompt})
        
        # 添加图片到内容中
        for image_name in image_names:
            image_path = os.path.join("images", image_name)
            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    # 将图片编码为base64
                    import base64
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
        
        print(content)
        
        # 调用OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature
        )
        res = completion.choices[0].message.content
        
        return res

class OpenAI_model_multimodal:
    def __init__(self,model_name:str='gpt-4o-mini',temperature:float=0.9, base_url:str=None) -> None:
        #初始化大模型
        self.model_name = model_name  # Store model_name
        self.temperature = temperature  # Store temperature
        if base_url:
            self.client = OpenAI(base_url=base_url) # Initialize official OpenAI client
        else:
            self.client = OpenAI() # Initialize official OpenAI client
        #加载向量数据库，embedding模型
        self.textdb=Vectordatabase()
        self.textdb.load_vector()
        self.imagedb=ImageVectordatabase()
        self.imagedb.load_vector()
        api_key = OpenAI.api_key
        self.embedding_model=Jinaembedding(api_key=api_key)

    def chat_multimodal(self,question:str):
        #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {info_text}
        {info_image}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
        info_text=self.textdb.query(question, self.embedding_model, 3)
        info_image=self.imagedb.query(question, self.embedding_model, 3)
        
        prompt=PromptTemplate(template=template, input_variables=["question","info"]).format(question=question, info_text=info_text, info_image=info_image)
        
        # 准备消息列表，首先添加系统提示
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 如果info中包含图片名称，从images文件夹加载图片
        image_names = [item[0] for item in info_image] if isinstance(info_image, list) and len(info_image) > 0 and isinstance(info_image[0], tuple) else []
        
        # 构建多模态消息
        content = []
        content.append({"type": "text", "text": prompt})
        
        # 添加图片到内容中
        for image_name in image_names:
            image_path = os.path.join("images", image_name)
            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    # 将图片编码为base64
                    import base64
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
        
        print(content)
        
        # 调用OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature
        )
        res = completion.choices[0].message.content
        
        return res
