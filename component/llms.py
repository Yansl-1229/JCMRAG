
from openai import OpenAI
from langchain.prompts import  PromptTemplate
from component.embedding import Zhipuembedding,Jinaembedding
from component.databases import Vectordatabase
import os
openai_api_key = os.getenv("OPENAI_API_KEY")

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
        api_key = openai_api_key
        self.embedding_model=Jinaembedding(api_key=api_key)
        
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

        info=self.db.query(question,self.embedding_model,1)

        prompt=PromptTemplate(template=template,input_variables=["question","info"]).format(question=question,info=info)

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

