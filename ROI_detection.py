import cv2
import numpy as np
from heatmappy import Heatmapper
from PIL import Image, ImageDraw

# For Report
def padding(img, width, height):
    delta_w = width - img.shape[1]
    delta_h = height - img.shape[0]
    result = cv2.copyMakeBorder(img,0,delta_h,0,delta_w,cv2.BORDER_CONSTANT)
    return result

# Set ROI
def find_position(points):
    sum_xy = np.sum(points, axis=1)
    diff_xy = np.diff(points, axis=1)
    
    topLeft = points[sum_xy.argmin()]
    bottomRight = points[sum_xy.argmax()]
    topRight = points[diff_xy.argmin()]
    bottomLeft = points[diff_xy.argmax()]
    
    print(topLeft,bottomRight,topRight,bottomLeft)
    return topLeft, topRight, bottomRight, bottomLeft

def output_size(topLeft, topRight, bottomRight, bottomLeft):
    w1, w2 = bottomRight[0] - bottomLeft[0], topRight[0] - topLeft[0]
    h1, h2 = bottomRight[1] - topRight[1], bottomLeft[1] - topLeft[1] 
    
    return int(max(w1, w2)), int(max(h1, h2))

def on_mouse(event, x, y, flags, param):
    global count, src, dst, width, height
    
    if event == cv2.EVENT_LBUTTONDOWN:       
        cv2.circle(draw_points, (x, y), 10, (0,0,255), 3)
        cv2.imshow('set points', draw_points)

        points[count] = (x, y)
        count += 1
        if count == 4:
            topL, topR, bottomR, bottomL = find_position(points)
            
            src = np.float32([topL, topR, bottomR, bottomL])
            cv2.polylines(draw_points, [np.array(src,np.int32)], True, (0,255,0), 5)
            
            width, height = output_size(topL, topR, bottomR, bottomL)

            dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            cv2.destroyAllWindows()


video_path = 'test_Trim.mp4'
output_path = './with_ROI_detection.mp4'

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

mask = np.zeros((1080,720))
event_occur = False

if video.isOpened():
    rv, frame = video.read()
    print("Frame Size:",frame.shape[1],frame.shape[0])
    while cv2.waitKey(1)<0:
        frame_num +=1
        event_occur = False
        return_value, frame = video.read()
        
        # 프레임 정보가 있으면 계속 진행 
        if return_value:
            pass
        else : 
            print('비디오가 끝났거나 오류가 있습니다')
            break
        frame = cv2.resize(frame,(1080, 720))
        h,w,c = frame.shape
        
        if(frame_num == 1):
            count = 0
            points = np.zeros((4, 2))
            draw_points = frame.copy()

            cv2.imshow('set points', frame)
            cv2.setMouseCallback('set points', on_mouse)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('Points',draw_points)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            colorCVT = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(colorCVT)
            imArray = np.asarray(im)

            maskIm = Image.new('L',(imArray.shape[1], imArray.shape[0]),0) # 마스크 탐지용
            maskIm_clear= Image.new('L',(imArray.shape[1], imArray.shape[0]),0) # 정상
            maskIm_fail= Image.new('L',(imArray.shape[1], imArray.shape[0]),0) # 탐지


            # 임시
            #src = [(282,94),(870,99),(671,628),(187,609)]
            ImageDraw.Draw(maskIm).polygon(src,outline=1,fill=1)
            mask = np.array(maskIm)

        newImArray = np.empty(frame.shape, dtype='uint8')
        newImArray[:,:,:3] = frame[:,:,:3]

        newImArray[:,:,0] = newImArray[:,:,0] * mask
        newImArray[:,:,1] = newImArray[:,:,1] * mask
        newImArray[:,:,2] = newImArray[:,:,2] * mask

        #crop_result = cv2.cvtColor(newImArray, cv2.COLOR_RGB2BGR)
        crop_result = newImArray


        # Mask Detection
        blob = cv2.dnn.blobFromImage(crop_result, 0.00392, (512,512),(0,0,0),True,crop=False)
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

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = [int(c) for c in COLORS[class_ids[i]]]

                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1)
                cv2.putText(frame, label, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

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

                    np.clip(ptz_loc[0],0,frame.shape[1])
                    np.clip(ptz_loc[1],0,frame.shape[0])

                    event_occur = True

        tracking_frame = frame[ptz_loc[1][0]:ptz_loc[1][1],ptz_loc[0][0]:ptz_loc[0][1]]
        ptz = padding(tracking_frame,200,250)

        ROI_frame = Image.fromarray(frame)
        display = ROI_frame.copy()
        drawing = ImageDraw.Draw(display, 'RGBA')
        if event_occur:
            drawing.polygon(src,outline=(0,0,255,255),fill=(0,0,255,50),width=3)
        else:
            drawing.polygon(src,outline=(255,0,0,255),fill=(255,0,0,50),width=3)
        display_result = np.array(display)

        # For Report
   
        #cv2.imshow('ptz',ptz)
        cv2.imshow('display_result', display_result)
        #cv2.imshow('With YOLO', bitwise_image)
        output_video.write(display_result)

video.release()
cv2.destroyAllWindows()