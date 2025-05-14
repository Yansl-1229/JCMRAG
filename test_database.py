from   component.embedding import Jinaembedding
from component.databases import Vectordatabase


api_key = "sk-VttLrtXYMsKnEs4CD01eA4D39575463486Ef5d7e2a063095"

text="项目结构"
embedding_model=Jinaembedding(api_key=api_key)

db=Vectordatabase()

db.load_vector()

result=db.query(text,embedding_model,1)

print(result)