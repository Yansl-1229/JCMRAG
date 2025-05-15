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
    
    def _encode_image_to_base64(self, image_path_or_url):
        """将图片路径或URL转换为base64编码"""
        import base64
        import os
        import requests
        from PIL import Image
        import io
        
        try:
            # 处理URL
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url)
                response.raise_for_status()
                image_content = response.content
            # 处理本地文件路径
            elif os.path.exists(image_path_or_url):
                with open(image_path_or_url, 'rb') as f:
                    image_content = f.read()
            else:
                print(f"Error: Image not found at {image_path_or_url}")
                return None
                
            # 转换为base64
            base64_encoded = base64.b64encode(image_content).decode('utf-8')
            return base64_encoded
            
        except Exception as e:
            print(f"Error encoding image to base64: {e}")
            return None
    
    def embed_images(self, image_paths_or_urls):
        payload_input = []
        
        # 确保image_paths_or_urls是列表
        if isinstance(image_paths_or_urls, str):
            image_paths_or_urls = [image_paths_or_urls]
            
        for img_ref in image_paths_or_urls:
            if isinstance(img_ref, str):
                # 检查是否已经是base64字符串
                if len(img_ref) > 100 and not img_ref.startswith(('http', '/', '\\')):  # 简单检查
                    payload_input.append({"image": img_ref})  # 假设已经是base64
                else:
                    # 转换为base64
                    base64_image = self._encode_image_to_base64(img_ref)
                    if base64_image:
                        payload_input.append({"image": base64_image})
            else:
                print(f"Warning: Skipping invalid image input: {img_ref}")
                
        if not payload_input:
            print("Error: No valid images to embed")
            return []
            
        data = {
            "model": self.embedding_model,
            "input": payload_input
        }
        
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        response.raise_for_status()
        results = response.json()
        return results["data"][0]["embedding"]
