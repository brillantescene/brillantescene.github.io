---
title:
categories: machine_learning
last_modified_at: 2020-09-07 18:29:51
tags: [AI, CNN, ML]
toc: true
toc_sticky: true
sidebar:
  nav: "docs"
---

# 내 데이터로 CNN 돌려보기

캡스톤 프로젝트

# 데이터 준비하기

---

우리는 6개 클래스를 가지고 있고 어쩌구 근데 지금은 3개 클래스에 대한 데이터만 잏ㅆ어서 일단 세개로만 해보겠음.

AWS Sagemaker를 이용할 계획인데, 그러려면 데이터셋을 pkl 파일로 만들어야 한다.

진짜 여기저기 삽질하다가 이 [깃헙](https://github.com/tikroute/mnist.pkl.gz-dataset-creator)을 발견하고 따라해봤다!

train, valid, test 폴더를 만들고 데이터를 7:2:1의 비율로 넣어주었다.

## 1. label이 들어간 csv 파일 만들어주기

csv 파일을 만들어주는 코드를 작성했다.

```python
import os
import natsort
import csv
import re

file_path = 'test/'
file_lists = os.listdir(file_path)
file_lists = natsort.natsorted(file_lists)

f = open('train.csv', 'w', encoding='utf-8') #valid.csv, test.csv
wr = csv.writer(f)
wr.writerow(["Img_name", "Class"])
for file_name in file_lists:
    print(file_name)
    wr.writerow([file_name, re.sub('-\d*[.]\w{3}', '', file_name)])
f.close()
```

이렇게 작성해서 실행해주면 간단하게 csv 파일 만들기 성공 〰️

`train.csv`를 열면 이렇게 생겼다!

![2020-08-284.03.46.png](/assets/images/posts/2020-09-07/2020-08-284.03.46.png)

## 2. 이미지 크기 조정하기

- 원래 이미지 크기는 310x770 사이즈였음
- 이걸 32x32로 바꿔보겠다!

**❗️어떤 크기가 적당한지 몰라서 일단 32, 32로 함❗️**

(바꿔보니 이미지가 너무 깨지나 싶어서 학습 진행하면서 조절할 예정)

![1-2.png](/assets/images/posts/2020-09-07/1-2.png)

- 깃헙 코드는 이미지 크기 변경을 Shell 스크립트로 했던데, 난 파이썬 코드로 만들었음

```python
from PIL import Image
import os
import natsort
import csv
import re

img_size = (32, 32)

def resize_img(img_path):
    img_lists = os.listdir(img_path)
    img_lists = natsort.natsorted(img_lists)
    for img_name in img_lists:
        print(f'{img_path}{img_name}')
        image = Image.open(f'{img_path}{img_name}')
        image = image.resize(img_size)
        image.save(f'{img_path}{img_name}')

resize_img('train/')
resize_img('valid/')
resize_img('test/')
```

- 이렇게 해서 32, 32 사이즈의 이미지로 변경!
- 콩알만 해졌다.

![drop-4.png](/assets/images/posts/2020-09-07/drop-4.png)

이제 이 데이터셋으로 피클링을 진행해보겠음!

## 3. pkl 파일로 만들기

깃헙 코드와 다른 점

- 원래는 grayscale로 변환했는데, 난 rgb 그대로 가지고 가겠다~!
- `python2`로 작성했는지 `cPickle`을 사용했는데, 난 `python3`이니까 `_pickle`을 사용함

```python
from PIL import Image
from numpy import genfromtxt
import gzip
import _pickle
from glob import glob
import numpy as np
import pandas as pd

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        img = Image.open(file_name)
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels, names = ["class"])
        return np.array(dataset), np.array(df["class"])
    else:
        return np.array(dataset)

Data1, y1 = dir_to_dataset("train/*.png","train.csv")
Data2, y2 = dir_to_dataset("valid/*.png","valid.csv")
Data3, y3 = dir_to_dataset("test/*.png","test.csv")

# Data and labels are read
train_num = 2758
valid_num = 844
test_num = 420

train_set_x = Data1[:train_num]
train_set_y = y1[1:train_num+1]
val_set_x = Data2[:valid_num]
val_set_y = y2[1:valid_num+1]
test_set_x = Data3[:test_num]
test_set_y = y3[1:test_num+1]

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('soundee.pkl.gz','wb')
_pickle.dump(dataset, f, protocol=2)
f.close()
```

# 데이터 로드하기

---

[https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)

이제 본격적으로 학습을 시작해보자!

- 먼저 데이터를 로드하고

  ```python
  %%time
  import _pickle, gzip, urllib.request, json
  import numpy as np

  with gzip.open('soundee.pkl.gz', 'rb') as f:
      train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
  ```

- 이미지와 레이블로 각각 나눠주고

  ```python
  (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = train_set, valid_set, test_set
  ```

- input 벡터

  ```python
  train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
  valid_images = valid_images.reshape(valid_images.shape[0], 32, 32, 3)
  test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

  train_images = train_images.astype('float32')
  valid_images = valid_images.astype('float32')
  test_images = test_images.astype('float32')
  ```

  ![2020-08-286.20.33.png](/assets/images/posts/2020-09-07/2020-08-286.20.33.png)

  ....

  **그래 웬일로 잘 나가나 했다^~^**

CIFAR-10 데이터셋을 받아서 데이터 shape을 출력해봄

![2020-08-286.32.19.png](/assets/images/posts/2020-09-07/2020-08-286.32.19.png)

이건 내 데이터

![2020-08-286.29.18.png](/assets/images/posts/2020-09-07/2020-08-286.29.18.png)

1024,,? 넌 왜,, 32\*32가 되어있니,,?

그리고 rgb 채널은 또 어따 팔아먹었어

어흑

pkl 파일 저장이 잘못된 것 같아서,,^^ 다시 뜯어보다가

```python
pixels = [f[0] for f in list(img.getdata())]
```

여기가 문제인 것 같은 삘이 왔다.

검색하다 [스택오버플로우](https://stackoverflow.com/questions/1109422/getting-list-of-pixel-values-from-pil)에서 `import imageio im = imageio.imread('sogreche.jpg')`

이 부분을 보고 소름 쫙 돋아서 모듈 다운받고 문서 찾아서 해봤다.

![2020-08-302.32.51.png](/assets/images/posts/2020-09-07/2020-08-302.32.51.png)

눈물나. 됐다.

다시 데이터 로드 해보고

![2020-08-302.34.12.png](/assets/images/posts/2020-09-07/2020-08-302.34.12.png)

떴다..

![2020-08-302.34.56.png](/assets/images/posts/2020-09-07/2020-08-302.34.56.png)

이미지도 잘 뜬다. 미쳤다

미 쳤 어

이제 다시 해보자,,,,

- input vector

  ```python
  train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
  valid_images = valid_images.reshape(valid_images.shape[0], 32, 32, 3)
  test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

  train_images = train_images.astype('float32')
  valid_images = valid_images.astype('float32')
  test_images = test_images.astype('float32')
  ```

  아아악 된다아악

  ~~아 기빨려,, 밥 먹고 올래~~ 먹고 옴!

- 이미지 픽셀 값 정규화 해주기

  ```python
  # normalizing the data to help with the training
  train_images /= 255
  valid_images /= 255
  test_images /= 255
  ```

- one-hot 인코딩 해주기

  ```python
  # one-hot encoding using keras' numpy-related utilities
  n_classes = 3
  print("Shape before one-hot encoding: ", train_labels.shape)
  train_labels = np_utils.to_categorical(train_labels, n_classes)
  valid_labels = np_utils.to_categorical(valid_labels, n_classes)
  test_labels = np_utils.to_categorical(test_labels, n_classes)
  print("Shape after one-hot encoding: ", train_labels.shape)
  ```

![2020-08-285.09.17.png](/assets/images/posts/2020-09-07/2020-08-285.09.17.png)

오류가 발생했다,,

읽어보니 label이 숫자여야 하나보다

그래서 [이 블로그](https://kwong3.tistory.com/6) 글을 보고 int로 바꿔줬다!

```python
from sklearn.preprocessing import LabelEncoder

def encoding_to_int(labels):
    e = LabelEncoder()
    e.fit(labels)
    return e.transform(labels)

n_classes = 3
print("Shape before one-hot encoding: ", train_labels.shape)
train_labels = np_utils.to_categorical(encoding_to_int(train_labels), n_classes)
valid_labels = np_utils.to_categorical(encoding_to_int(valid_labels), n_classes)
test_labels = np_utils.to_categorical(encoding_to_int(test_labels), n_classes)
print("Shape after one-hot encoding: ", train_labels.shape)
```

![2020-08-286.06.11.png](/assets/images/posts/2020-09-07/2020-08-286.06.11.png)

음. 잘 돌아가는군! 이제 다음 〰️

# 모델 빌드 & 학습하기

---

우리의 최종 목표는 resnet으로 돌리는 건데, 일단 테스트용으로 [이 블로그](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)에서 sequential API로 만든 cnn 모델 그대로 써주겠음!

```python
 # building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(3, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
```

테스트니까 간단쓰하게 10epoch만 돌려보기 ~,~

```python
# training the model for 10 epochs
model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_data=(valid_images, valid_labels))
```

![2020-08-304.56.52.png](/assets/images/posts/2020-09-07/2020-08-304.56.52.png)

음 결과가 아주 별로다. 괜찮아

# 정확도 평가하기

---

test_set으로 정확도 평가하기~!

```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)
```

![2020-08-304.58.44.png](/assets/images/posts/2020-09-07/2020-08-304.58.44.png)

와우 테스트 정확도 진짜 난리났다. 괜찮다

# 예측만들기

---

```python
prediction = model.predict(test_images)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
```

간단하게만 볼 거라 예측 레이블을 아무렇게나 출력해보겠다!

```python
cnt = 0
for i in prediction:
    pre_ans = i.argmax()
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "drop"
    elif pre_ans == 1: pre_ans_str = "motor"
    elif pre_ans == 2: pre_ans_str = "water"


    if i[0] >= 0.4 : print(f"{test_labels[cnt]} 이미지는 {pre_ans_str}로 추정됩니다.")
    if i[1] >= 0.4: print(f"{test_labels[cnt]} 이미지는 {pre_ans_str}로 추정됩니다.")
    if i[2] >= 0.4: print(f"{test_labels[cnt]} 이미지는 {pre_ans_str}로 추정됩니다.")
    if i[0] < 0.4 and i[1] < 0.4 and i[2] < 0.4 : print(f"{test_labels[i]} 이미지를 추정할 수 없습니다.")
    print()
    cnt += 1
```

![2020-08-305.04.22.png](/assets/images/posts/2020-09-07/2020-08-305.04.22.png)

좋아쓰. 기본 cnn 돌아가는거 확인했으니 이제

- resnet으로 학습하고
- 최적 모델 저장하고
- 모델 불러 와서 예측
- 기계학습 모형 배포

이렇게 해보겠다.

우리는 실시간으로 들어오는 소리 데이터를 입력으로 해서 예측 모델에 넣고 예측값 받고, 그걸 애플리케이션에 넘겨줘야 한다.

이걸 sagemaker에서 해야 하는데,,, 미친 안된다. 울고싶다. 그래도 해야하니 해보겠다..^^

이번 포스팅 끝 ~,~

다음 포스팅 👉 [내 데이터로 ResNet 돌려보기](https://www.notion.so/ResNet-d5c9229fc28a46659984c445fb68cfdb)
