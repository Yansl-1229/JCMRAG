import numpy as np
from numpy.linalg import norm
from zhipuai import ZhipuAI
from typing import List
import requests
import json
#从系统变量中获取api_key
import os
zhipu_api_key = os.getenv("ZhipuAI_API_KEY")

# ZhipuAI 嵌入
class Zhipuembedding:

    def __init__(self, path:str=''):
        	

        client = ZhipuAI(api_key=zhipu_api_key) 
        self.embedding_model=client


    def get_embedding(self,content:str=''):
        response =self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=content #填写需要计算的文本内容,
        )
        return response.data[0].embedding

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
    def compare(self, text1: str, text2: str):
        embed1=self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=text1 #填写需要计算的文本内容,
        ).data[0].embedding

        embed2=self.embedding_model.embeddings.create(
            model="embedding-2", #填写需要调用的模型名称
            input=text2 #填写需要计算的文本内容,
        ).data[0].embedding

        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    

# JinaAI 嵌入
class Jinaembedding:
    def __init__(self, api_key):
        self.url = 'https://aihubmix.com/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.embedding_model = "jina-clip-v2"

    def embed_texts(self, texts):
        payload_input = []
        for text in texts:
            payload_input.append({"text": text})

        data = {
            "model": self.embedding_model,
            "input": payload_input
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        response.raise_for_status() # Raise an exception for HTTP errors
        results = response.json()
        return [item["embedding"] for item in results["data"]]

    
    def get_embedding(self,content:str=''):
        payload_input = [{"text": content}]
        data = {
            "model": self.embedding_model,
            "input": payload_input
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        response.raise_for_status()
        results = response.json()
        return results["data"][0]["embedding"]

    def compare(self, text1: str, text2: str):
        
        cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        embeddings = self.embedding_model.encode([text1, text2])
        return cos_sim(embeddings[0], embeddings[1])

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
