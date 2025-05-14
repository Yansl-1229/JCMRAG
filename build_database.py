from openai import api_key
from requests import api
from   component.embedding import Zhipuembedding,Jinaembedding
from component.data_chunker import ReadFile
from component.databases import Vectordatabase

filter=ReadFile('./data')
docs=filter.get_all_chunk_content(200,150)

docs

# print(docs.Type)

api_key = "sk-VttLrtXYMsKnEs4CD01eA4D39575463486Ef5d7e2a063095" 

embedding_model=Jinaembedding(api_key=api_key)

database=Vectordatabase(docs)

Vectors=database.get_vector(embedding_model)

database.persist()