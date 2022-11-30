import cv2
import numpy as np
from heatmappy import Heatmapper
from PIL import Image

# For Report
def padding(img, width, height):
    delta_w = width - img.shape[1]
    delta_h = height - img.shape[0]
    result = cv2.copyMakeBorder(img,0,delta_h,0,delta_w,cv2.BORDER_CONSTANT)
    return result

video_path = 'ROI_test.mp4'
output_path = './background_subtraction_output.mp4'

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'DIVX')
output_video = cv2.VideoWriter(output_path, codec, fps, (1080, 720)) 

# 배경과 객체 구분을 위한 함수
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=3,backgroundRatio=0.7, noiseSigma=0)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=16,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.5)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
close_kernel = np.ones((5,5), np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

frame_num = 0

# YOLO 사용
YOLO_net = cv2.dnn.readNet("data/yolov3_mask_last.weights", "data/detect_mask.cfg")
YOLO_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
YOLO_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
COLORS = [[0,255,0],[0,0,0],[0,0,255]]

with open("data/object.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]
ptz_loc = np.array([[0,0],[0,0]])
tracking_frame = np.zeros((940,540))
heatmap_frame = np.zeros((1080,720))
heatmap_loc_stack = []

if video.isOpened():
    rv, frame = video.read()
    print("Frame Size:",frame.shape[1],frame.shape[0])
    while cv2.waitKey(1)<0:
        frame_num +=1         
        return_value, frame = video.read()
        
        # 프레임 정보가 있으면 계속 진행 
        if return_value:
            pass
        else : 
            print('비디오가 끝났거나 오류가 있습니다')
            break
        frame = cv2.resize(frame,(1080, 720))
        h,w,c = frame.shape
        
        background_extraction_mask = fgbg.apply(frame)
        background_extraction_mask = cv2.morphologyEx(background_extraction_mask, cv2.MORPH_OPEN, kernel)
        background_extraction_mask = cv2.morphologyEx(background_extraction_mask, cv2.MORPH_CLOSE, close_kernel)
        #background_extraction_mask = cv2.dilate(background_extraction_mask,kernel,iterations=1)
        background_extraction_mask = np.stack((background_extraction_mask,)*3, axis=-1)
        background_inverse = cv2.bitwise_not(background_extraction_mask)

        bitwise_image = cv2.bitwise_and(frame, background_extraction_mask)
        background_image = cv2.bitwise_and(frame, background_inverse)
        #background_image = cv2.GaussianBlur(background_image, (0,0), 1)

        #concat_image = np.concatenate((frame,bitwise_image), axis=1)
        background_concat_image = np.concatenate((background_inverse, background_image), axis=1)
        
        # Mask Detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (512,512),(0,0,0),True,crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)

                    x = int(center_x - dw/2)
                    y = int(center_y - dh/2)
                    boxes.append([x,y,dw,dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

        result = bitwise_image + background_image

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = [int(c) for c in COLORS[class_ids[i]]]

                cv2.rectangle(result, (x,y), (x+w,y+h), color, 1)
                cv2.putText(result, label, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

                if label == 'no mask':
                    print("No Mask location:", x, y)
                    print("BOX SIZE:",w,h)
                    ptz_loc[0][0] = x - 100
                    ptz_loc[0][1] = x +  100
                    ptz_loc[1][0] = y - 50
                    ptz_loc[1][1] = y + 200
                    ptz_loc = np.uint32(ptz_loc)

                    # 히트맵을 위한 가중치 저장
                    heatmap_loc_stack.append((x,y))
                    print(heatmap_loc_stack)

                    np.clip(ptz_loc[0],0,bitwise_image.shape[1])
                    np.clip(ptz_loc[1],0,bitwise_image.shape[0])

        tracking_frame = frame[ptz_loc[1][0]:ptz_loc[1][1],ptz_loc[0][0]:ptz_loc[0][1]]
        ptz = padding(tracking_frame,200,250)


        heatmapper = Heatmapper(
            point_diameter = 150,  # the size of each point to be drawn
            point_strength = 0.02,  # the strength, between 0 and 1, of each point to be drawn
            opacity = 0.8,  # the opacity of the heatmap layer
            colours = 'default',  # 'default' or 'reveal'
                                # OR a matplotlib LinearSegmentedColorMap object 
                                # OR the path to a horizontal scale image
            grey_heatmapper = 'PIL'  # The object responsible for drawing the points
                                # Pillow used by default, 'PySide' option available if installed
        )

        result_convert = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV BGR to PIL RGB
        numpy_frame = np.array(result_convert) # 배열로 전환
        heatmap_frame = Image.fromarray(numpy_frame,'RGB') # PIL 형식으로 전환
        heatmap = heatmapper.heatmap_on_img(heatmap_loc_stack, heatmap_frame) # 히트맵 그리기
        numpy_heatmap_frame = np.array(heatmap) # 배열로 전환
        heatmap_result = cv2.cvtColor(numpy_heatmap_frame, cv2.COLOR_RGB2BGR) # PIL RGB to OpenCV BGR, OpenCV 형식으로 전환

        # For Report
        #third_camera = frame[360:,:540]

        #cv2.imshow('mask', background_concat_image)       
        cv2.imshow('result',result)
        #cv2.imshow('heatmap',heatmap_result)
        #cv2.imshow('ptz',ptz)
        #cv2.imshow('3 camera', third_camera)
        output_video.write(result)

video.release()
cv2.destroyAllWindows()