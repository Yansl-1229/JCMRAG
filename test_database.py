from component.embedding import Jinaembedding
from component.databases import Vectordatabase
import os
api_key = os.getenv("OPENAI_API_KEY")


text="项目结构"
embedding_model=Jinaembedding(api_key=api_key)

db=Vectordatabase()

db.load_vector()

result=db.query(text,embedding_model,1)

print(result)