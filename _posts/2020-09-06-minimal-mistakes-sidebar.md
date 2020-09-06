---
title: "minimal mistakes 테마에서 sidebar 만들기"
categories:
  - blogging
last_modified_at: 2020-09-06T13:00:00+09:00
toc: true
#sidebar:
#  nav: "docs"
---

[minimal mistakes 문서](https://mmistakes.github.io/minimal-mistakes/docs/layouts/#custom-sidebar-content)를 참고하여 쓴 글 입니다.

먼저 테스트 용으로 문서에 나와있는 그대로 복사에서 이 포스트의 YAML Front Matter에 붙여넣기 해줬다.
(사진)
오, 사이드로 나왔다.

하지만 저 문서처럼 사이드 바를 만들고 싶기 때문에 **custom-sidebar-content**를 따라 해보겠다.

구성하고 싶은 사이드 바 카테고리는

- Machine Learning
- Algorithm
- Server
- iOS
- Python
  이 정도로 해놓겠다.

notion에 정리했던 노트들을 이 블로그에 하나씩 가져올 거라 미리 `children`에 써주겠다.

```
docs:
  - title: Machine Learning
    children:
      - title: "내 데이터로 CNN 돌려보기"
        url: /_pages/cnn/
      - title: "내 데이터로 ResNet 돌려보기"
        url: /_pages/resnet/
      - title: "AWS Sagemaker에서 ResNet 돌려보기"
        url: /_pages/sagemaker/

  - title: Algorithm
    children:
      - title: "Configuration"
        url: /docs/configuration/

  - title: Server
    children:
      - title: "Configuration"
        url: /docs/configuration/

  - title: iOS
    children:
      - title: "[Swift] 네비게이션 바, 화면 이동"
        url: /_pages/swift1/
      - title: "[Swift] 웹뷰(WKWebView), 옵셔널 바인딩(optional binding)"
        url: /_pages/swift2/

  - title: Python
    children:
      - title: "Configuration"
        url: /docs/configuration/
```

이렇게 포스트를 전부 다 사이드 바에 표시하면 나중에 지저분해지겠지만 지금은 사이드 바 만드는 게 목표니까 그냥 해주겠다.
