# 마스크 탐지 지능형 CCTV
#### 2017103962 김동규

## 시스템 설명
Main Camera는 Wide-View로 촬영하면서 감시 구역을 계속 촬영하고 마스크 탐지 기능을 추가한다. 마스크 미착용 인원이 감지될 경우, PTZ Sub-Camera가 미착용 인원을 추적하여 사람의 상시 감시 없이 자동으로 대응하는 시스템이다.

### 기능
* Wide-View Angle 영상
* 마스크 미착용자 Zoom-in, Tracking
* 마스크 미착용자 빈도 수 히트맵 영상
* 관심 구역 지정

### 처리 순서
1. 원본 영상에서 마스크로 객체와 배경 분리

<img src="demo/mask.png"></img>


2. 객체 영상에서 마스크 탐지 중 미착용 인원 발견 시 좌표 전달

<img src="demo/ptz.png"></img>


3. PTZ 카메라로 추적

<img src="demo/tracking.png"></img>


4. Wide-angle view는 객체+배경 영상으로 원본과 동일

### 구조
<img src="demo/arch.JPG"></img>


### Requirements
* yolov3, yolov4
* matplotlib
* Pillow
* numpy
* OpenCV

## How to use

### 1. Mask Detection
--------------------
#### 1.1. Load weights
* Use settings/training.ipynb
~~~
!wget https://pjreddie.com/media/files/darknet53.conv.74
~~~

or Download [weights](https://drive.google.com/file/d/1_TOW4zOeoOkBm5hWePTDCmYxOKNxzgHu/view?usp=share_link)

#### 1.2. Train
~~~
!./darknet detector train /content/Face_Mask_Detection_YOLO/Mask/object.data\
                          /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                          darknet53.conv.74\
                          -dont_show -map 
~~~

#### 1.3. Detect Examples
~~~
!./darknet detector test /content/Face_Mask_Detection_YOLO/Mask/object.data\
                         /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                         /content/backup/detect_mask_last.weights\
                         /content/Face_Mask_Detection_YOLO/demo/man_0_1.png
~~~

### 2. Person Detection
-----------------------
Download Tiny YOLO cfg, weights at <https://pjreddie.com/darknet/yolo/>

### 3. Install Heatmap settings
---------------------
~~~
python heatmap/setup.py install
~~~

## About Models

Dataset
-----------------

<img src="demo/yolo_train.png"></img>

Class
* **mask** : 마스크 착용
* **improperly** : 턱 밑으로 내린 마스크
* **no mask** : 마스크 미착용

with yolov4
|Set|Images|with Mask|without Mask|
|:-----:|:-----:|:-----:|:-----:|
|Training|700|3047|868|
|Validation|100|278|49|
|Test|120|503|156|
|Total|920|3828|1073|

결과
|모델명|Training Set|Validation Set|Test Set|
|:-----:|:-----:|:-----:|:-----:|
|YOLO v4|99.65%|88.38%|93.95%|

Image Sources
--------------
* Dataset: <https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG>