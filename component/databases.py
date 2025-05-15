from tqdm import tqdm
import numpy as np 
import os
import json
from typing import List, Tuple # Added Tuple for type hinting


#定义一个向量数据库类
class Vectordatabase:
    
    #初始化方法，传入一个字块列表
    def __init__(self,docs:List=[]) -> None:
        self.docs = docs
    
    #对字块列表进行，批量的embedded编码，传入embedding模型，返回一个向量列表
    def get_vector(self,EmbeddingModel)->List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.docs):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    
    #把向量列表存储到json文件中，把子块列表也存储到json文件,默认路径为'database'
    def persist(self,path:str='database')->None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    #加载json文件中的向量和字块，得到向量列表、字块列表,默认路径为'database'
    def load_vector(self,path:str='database')->None:
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
    
    #求向量的余弦相似度，传入两个向量和一个embedding模型，返回一个相似度
    def get_similarity(self, vector1: List[float], vector2: List[float], embedding_model) -> float:
        return embedding_model.compare_v(vector1, vector2)
    
    #求一个字符串和向量列表里的所有向量的相似度，表进行排序，返回相似度前k个的子块列表
    def query(self, query: str, EmbeddingModel, k: int = 3) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector, EmbeddingModel)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()


class ImageVectordatabase:
    # 传入一个图片路径列表
    def __init__(self,path_list:List[str]=[]) -> None: # path_list elements are strings (paths)
        self.path_list = path_list
        self.image_embeddings: List[Tuple[str, List[float]]] = [] # Initialize image_embeddings
    
    # 对图片路径列表进行批量的embedded编码，传入embedding模型，
    # 生成 (imagename, imageembedding) 格式的数据列表
    def get_vector(self, EmbeddingModel) -> List[Tuple[str, List[float]]]:
        self.image_embeddings = [] # Clear previous results
        for path in tqdm(self.path_list):
            image_name = os.path.basename(path) # Extract image name from path
            embedding = EmbeddingModel.embed_images(path)
            self.image_embeddings.append((image_name, embedding))
        return self.image_embeddings

    # 把 (imagename, imageembedding) 列表存储到json文件中, 默认路径为'database'
    def persist(self, dir_path: str = 'database', filename: str = 'image_data.json') -> None:
        if not self.image_embeddings:
            print("警告: 没有图像嵌入数据可供持久化。请先调用 get_vector() 方法。")
            return
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            # JSON不支持元组，它们会被转换成列表
            json.dump(self.image_embeddings, f, ensure_ascii=False, indent=4)
        print(f"图像嵌入数据已保存到 {file_path}")

    # 从json文件中加载 (imagename, imageembedding) 数据, 默认路径为'database'
    def load_vector(self, dir_path: str = 'database', filename: str = 'image_data.json') -> None:
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。")
            self.image_embeddings = []
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            # 确保加载的数据是 [name, embedding] 列表的列表，并转换回元组列表
            self.image_embeddings = []
            for item in loaded_data:
                if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list):
                    self.image_embeddings.append((item[0], item[1]))
                else:
                    print(f"警告: 在 {file_path} 中发现格式不正确的数据项: {item}")
            print(f"图像嵌入数据已从 {file_path} 加载。")

    # (可选) 未来可以添加基于图像嵌入的查询方法
    def query(self, query_image_path: str, EmbeddingModel, k: int = 3) -> List[Tuple[str, float]]:
        if not self.image_embeddings:
            print("错误: 图像嵌入数据库为空。请先加载数据或生成嵌入。")
            return []
        query_embedding = EmbeddingModel.get_embedding(query_image_path)
        similarities = []
        for name, embedding in self.image_embeddings:
            # 假设 EmbeddingModel 有一个 compare_v 方法用于比较向量
            # 或者您需要在这里实现余弦相似度计算
            similarity = EmbeddingModel.compare_v(query_embedding, embedding) 
            similarities.append((name, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]