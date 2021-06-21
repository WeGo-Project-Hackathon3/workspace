# [ 자율주행 Follow Me 로봇 ]



## 1. 프로젝트 개요

- 프로젝트명 : 자율주행 Follow Me 로봇

- 기업명 : WeGo

- 팀명 : NuGo / 팀원 : 송용호, 김진오, 박정은, 이상미, 주진경

- 프로젝트 기간 : 2021.05.10 ~ 2021. 06.23 (약 2개월)
- 프로젝트 상세 : SSACxAIFFEL 인공지능 혁신학교 Hackathon3



## 2. 프로젝트 목표 및 요건

최근 자율주행 로봇과 관련한 서비스들이 출시되며 서빙, 순찰, 방역 등의 역할을 대체하고 있음
딥러닝 기술을 적용하여 추종하는 Follow Me 로봇을 개발하여 이러한 니즈를 충족하고자 함

지정된 사용자를 따라 물건을 싣고 이동하는 로봇
수행 요건

1. 비전 처리를 통해 사람의 모습, 형태를 학습하여 구분 및 인지
2. Depth 카메라 or LiDar 센서를 사용하여 일정 거리를 유지할 수 있도록 측정
3. 추적 중간에 방해(대상의 이탈, 새로운 대상의 유입)가 발생하더라도 지정된 대상만 Follow  



## 3. 프로젝트 프로세스

<img src="./images/image_02.png" alt="스크린샷, 2021-06-21 16-26-07" style="zoom:80%;" />



## 4. 알고리즘 개요

<img src="./images/image_01.png" alt="스크린샷, 2021-06-21 16-30-31" style="zoom:80%;" />



## 5. 프로젝트 환경구성

- 주요 환경 구성은 아래와 같다. 

  Ubuntu 18.4, anaconda, pyrealsense2, opencv-contrib-python, opencv-python, ros 등

- pip install 로 환경 구성 

```
$ pip install -r requirements.txt  
```

  

## 6. 프로젝트 실행 방법

스크립트 경로 : 워크스페이스의 /src/scout_mini_ros/scout_bringup 폴더 내에 존재 

```
$ python3  cv_tracking_hand_depth_cam.py
```

수정 중 ...    



## 7. References 

https://arxiv.org/abs/2004.10934

https://github.com/AlexeyAB/darknet

https://github.com/theAIGuysCode/yolov4-deepsort

https://github.com/theAIGuysCode/tensorflow-yolov4-tflite

추가 중 ...     



