from component.embedding import Jinaembedding
from component.databases import Vectordatabase,ImageVectordatabase
import os
api_key = os.getenv("OPENAI_API_KEY")


text="What is the cycle life of this 3.2V 280ah Lifepo4 battery?"
embedding_model=Jinaembedding(api_key=api_key)

db=ImageVectordatabase()

db.load_vector()

result=db.query(text,embedding_model,3)

print(result)