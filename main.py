import tensorflow as tf
import numpy as np
import pandas as pd

# csv데이터 load
data = pd.read_csv('/Users/hanseol/Downloads/gpascore.csv')

# print(data)  # data 출력
# data.isnull().sum()  # 데이터 내부 빈칸 찾기
data = data.dropna()  # 빈칸/NAN행 제거
# data.fillna(100)  # 데이터 빈칸 괄호값으로 변환
# print(data['gpa'])  # 데이터 특정 열의 데이터만 출력
# print(data['gpa'].min())  # 데이터 특정 열의 최솟값 출력 (max()는 최댓값)
# print(data['gpa'].count())  # 데이터 특정 열의 갯수 출력

# data 전처리
y_data = data['admit'].values  # admit열 값 가져오기

x_data = []
for i, rows in data.iterrows():  # iterrow(): data라는 dataframe내 데이터를 한 행(가로)씩 출력하기 i는 카운트 row에는 데이터 입력
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

# 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),  # 일반적인 노드(노드의 갯수, activation=유형) hidden layer
    tf.keras.layers.Dense(128, activation='tanh'),  # 일반적인 노드(노드의 갯수, activation=유형) hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid'),  # 일반적인 노드(노드의 갯수) output 마지막 노드이므로 결과물이 나와야 하기 때문에 1개
])

# 모델 컴파일(최적화func, lossfunc, 평가요소)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습(학습데이터(numpyarray or tensor), 학습데이터에 대응되는 값(numpyarray or tensor), epochs=학습횟수)
model.fit(np.array(x_data), np.array(y_data), epochs=7000)

# 모델을 통한 예측
result = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(result)
