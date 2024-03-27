#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:


data = pd.read_csv('data.csv')
data = data.dropna()


# In[ ]:


# 데이터 전처리
X = data['제목'] + ' ' + data['내용']
y = data['label']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=100)

# LightGBM 모델 훈련
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)

# 모델 평가 및 임계값 설정
y_pred_prob = lgbm_model.predict_proba(X_test)
y_pred_custom = [1 if prob[1] > 0.525 else 2 if prob[1] >= 0.475 else 0 for prob in y_pred_prob]

# Classification Report 출력
class_report_custom = classification_report(y_test, y_pred_custom)
print("Modified Classification Report:\n", class_report_custom)

# 모델 예측
y_pred = lgbm_model.predict(X_test)

# 전체 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("전체 정확도:", accuracy)


# In[2]:


import pickle

# 모델과 벡터라이저를 하나의 딕셔너리 객체로 묶음
model_and_vectorizer = {'model': lgbm_model, 'vectorizer': tfidf_vectorizer}

# 딕셔너리 객체를 파일로 저장
with open("model_and_vectorizer.dump", "wb") as fw:
    pickle.dump(model_vectorizer, fw)


# In[ ]:




