from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import  cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd

xlsx = pd.read_excel("text_summary_solution.xlsx")
# print(type(xlsx['one'][0]))

sentences = []
# for i in range(10):
#     sentences.append(xlsx['summary'][i])
#     sentences.append(xlsx['one'][i])
#     sentences.append(xlsx['two'][i])
#     sentences.append(xlsx['three'][i])
#     sentences.append(xlsx['four'][i])
#     sentences.append(xlsx['five'][i])
sentences.append(xlsx['summary'][0])
sentences.append(xlsx['one'][0])
sentences.append(xlsx['two'][0])
sentences.append(xlsx['three'][0])
sentences.append(xlsx['four'][0])
sentences.append(xlsx['five'][0])

# print(sentences)
# 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# 문장 벡터화 진행
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# 각 단어
text = tfidf_vectorizer.get_feature_names()

# 각 단어의 벡터 값
idf = tfidf_vectorizer.idf_

# print(dict(zip(text, idf)))

# 코사인 유사도
# print(tfidf_matrix[0:1])
# print(tfidf_matrix[1:2])
# print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))

# 유클리디언 유사도
print("유클리디언 유사도")
def l1_normalize(v):
    norm = np.sum(v)
    return v/norm

tfidf_norm_l1 = l1_normalize(tfidf_matrix)
max = 0
answer = 0
for i in range(5):
    temp = euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[i+1:i+2])
    print("temp : %f" % temp)
    if(max < temp):
        max = temp
        answer = i+1

print("실제 답 : %d , 예측 답 : %d" % (xlsx['solution'][0], answer))
if xlsx['solution'][0] == answer:
    print("정답입니다.")
else:
    print("틀렸습니다.")
# print(euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2]))