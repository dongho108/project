# Deep Learning Summarization


## 사용법

### 환경세팅
1. 제 파일들을 전부 clone 해주세요
2. import가 필요한 pakage들을 install 해주세요.
3. 데이터 다운 : ```https://www.kaggle.com/snap/amazon-fine-food-reviews``` 에서 ```reviews.csv``` 를 다운받아주세요.
4. attention.py를 다운받아주세요
> 1) ```https://raw.githubusercontent.com/thushv89/attention_keras/master/src/layers/attention.py```에서 직접 다운받거나  
> 2) ```urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/src/layers/attention.py", filename="attention.py")```를 통해 다운받은후  

```from attention import AttentionLayer``` import문을 추가해주세요


### training된 모델을 이용해 텍스트 요약해보기
1. attention이 미적용된 파일은 seq2seq_run.py를, attention이 적용된 파일은 Attention_run.py를 열어주세요.
2. 100000, 200000개의 데이터가 학습된 모델이 이미 준비되어 있습니다. 3번을 통해 변수값을 바꿔주시고 바로 이용해주시면 됩니다.
3. 원하는 모델을 사용하기 위해 numData에는 몇개의 데이터를 학습시킨 모델을 사용할건지, text_max_len에는 최대길이를 몇으로 제한한 모델을 사용할건지 입력해줍니다.
```
numData = 200000
name_Model = "seq2seq_Model_"+str(numData)
name_encoder_model = "seq2seq_encoder_Model_"+str(numData)
name_decoder_model = "seq2seq_decoder_Model_"+str(numData)
text_max_len = 50
```
4. (참고)test할 데이터에도 전처리가 필요하기 때문에 training시 사용한 전처리과정을 그대로 사용합니다.
5. run 해주시면 알아서 진행됩니다! (아래는 출력 예시)
```
원문 :  enjoying nutiva certified organic extra virgin coconut oil moderate low carb food plan fits right healthy diet use making low carb chocolate candy low carb chocolate cake spread use moisturizer skin also thought good value money completely satisfied 
실제 요약문 : completely satisfied 
예측 요약문 :  delicious
유클리디언 유사도 :  [[0.58578644]]


원문 :  love coconut oil use face body everyday night love smell also coconutty 
실제 요약문 : love it 
예측 요약문 :  great product
유클리디언 유사도 :  [[0.5]]


원문 :  really like oil coconut oil good oils touted good oil set use skin well 
실제 요약문 : organic coconut oil 
예측 요약문 :  great product
유클리디언 유사도 :  [[0.44948974]]


유클리디언 유사도 평균 :  [[0.40521263]]
```

### 모델 훈련
1. 새 모델을 생성 하시려면 ```seq2seqTrainig.py``` 혹은 ```AttentionTraining.py``` 에서 numData에는 학습시킬 데이터의 수를, text_max_len 에는 원문길이를 몇 개로 제한할 지 입력해주세요. (원문길이는 version 1.2 최대 50까지 됩니다.)
```
numData = 200000
text_max_len = 50

name_Model = "seq2seq_Model_"+str(numData)
name_encoder_model = "seq2seq_encoder_Model_"+str(numData)
name_decoder_model = "seq2seq_decoder_Model_"+str(numData)
```

2. run을 해주세요!
3. 다음과 같은 출력이 뜨면 됩니다.
```
모델 생성 완료
training seq2seq : seq2seq_Model_200000
test encoder : seq2seq_encoder_Model_200000
test decoder : seq2seq_decoder_Model_200000
```
4. ```seq2seq_Model_{데이터수}, seq2seq_encoder_Model_{데이터수}, seq2seq_decoder_Model_{데이터수}```의 이름을 가진 모델들이 생성됩니다. (적용된 모델도 마찬가지)

