from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk


sentry_sdk.init(
    dsn="https://b39662bf227e7a1a12555a8ce9cbd840@o4506983382188032.ingest.us.sentry.io/4506983385202688",
    environment="dev"
)
app = FastAPI(title="ML API")
app.add_middleware(SentryAsgiMiddleware)


with open("model_and_vectorizer.dump", "rb")as f:
    loaded_data = pickle.load(f)


# 불러온 데이터에서 모델과 벡터라이저를 각각 추출
loaded_model = loaded_data['model']
loaded_vectorizer = loaded_data['vectorizer']


class Politics(BaseModel):
    title: str
    content: str

# 새 데이터의 '제목'과 '내용'을 결합

@app.post("/predict", status_code=200)
async def predict_tf(x: Politics):
    # '제목'과 '내용'을 하나의 문자열로 결합하여 리스트에 넣은 후 DataFrame 생성
    new_data = pd.DataFrame({"data": [x.title + ' ' + x.content]})
    X_new_tfidf = loaded_vectorizer.transform(new_data['data'])
    res = loaded_model.predict(X_new_tfidf)
    percentage = loaded_model.predict_proba(X_new_tfidf)[0]
    percentage0 = percentage[0]
    percentage1 = percentage[1] 
    
    return {"prediction": res.tolist(), "percentage0": percentage0, "percentage1": percentage1}


@app.get('/')
async def root():
    return {"message": "online"}


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0