from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel

with open("model_and_vectorizer.dump", "rb")as f:
    loaded_data = pickle.load(f)


# 불러온 데이터에서 모델과 벡터라이저를 각각 추출
loaded_model = loaded_data['model']
loaded_vectorizer = loaded_data['vectorizer']

app = FastAPI(title="ML API")

class Politics(BaseModel):
    title: str
    content: str

# 새 데이터의 '제목'과 '내용'을 결합



@app.post("/predict", status_code=200)
async def predict_tf(x: Politics):
    new_data = pd.DataFrame({"제목": [x.title], "내용": [x.content]})
    X_new = new_data['제목'] + ' ' + new_data['내용']
    X_new_tfidf = loaded_vectorizer.transform(X_new)
    res= loaded_model.predict(X_new_tfidf)
    
    return {"prediction": res}

@app.get('/')
async def root():
    return {"message": "online"}