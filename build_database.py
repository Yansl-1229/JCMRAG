import openai
from component.embedding import Zhipuembedding,Jinaembedding
from component.data_chunker import ReadFile,ReadImages
from component.databases import Vectordatabase,ImageVectordatabase
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
zhipu_api_key = os.getenv("ZhipuAI_API_KEY")

print(zhipu_api_key)
print(openai.api_key)


# filter=ReadFile('./data')
# docs=filter.get_all_chunk_content(200,150)

# docs

# embedding_model=Jinaembedding(api_key=openai.api_key)

# database=Vectordatabase(docs)

# Vectors=database.get_vector(embedding_model)

# database.persist()

imgfilter = ReadImages('./images')

imglist = imgfilter.readlist()

print(imglist)

embedding_model=Jinaembedding(api_key=openai.api_key)

image_database=ImageVectordatabase(imglist)

Vectors=image_database.get_vector(embedding_model)

image_database.persist()