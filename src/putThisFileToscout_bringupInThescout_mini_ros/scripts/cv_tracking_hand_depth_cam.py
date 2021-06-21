#! /home/ssac23/anaconda3/envs/hack3/bin/python
import time
import os
import tensorflow as tf
import core.utils as utils
from core.config import cfg

from tensorflow.python.saved_model import tag_constants

import cv2
import numpy as np

import matplotlib.pyplot as plt
from hand_depth import hand_detection, hand_init

# ROS
import rospy
from geometry_msgs.msg import Twist

# depth camera 
import pyrealsense2 as rs
from realsense_depth import *

#절대경로 설정
absolute_path = os.path.dirname(os.path.abspath(__file__))

#model_zoo에서 갈아끼울 모델 path 설정
MODEL_PATH = os.path.join(absolute_path, 'checkpoints/yolov4-tiny-416_50000')

#MODEL_PATH = os.path.join(absolute_path, 'checkpoints/yolov4-tiny-416')
IOU_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.35
INPUT_SIZE = 416
algo = "kcf"

#success = True
#DETECT_AFTER = 100
x,y,w,h = 0,0,0,0

# hand_detection 고려해야함
init_check = True

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
print("MODEL_PATH : ", MODEL_PATH)

# kcf :
# csrt :
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

#iou_구하기
def get_iou(boxA, boxB):
	""" Find iou of detection and tracking boxes
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)

	# return the intersection over union value
	return iou

def get_coordinates(box, x, y, x1, y1):
	""" Get co-ordinates of flaged person
	"""
	if len(box) == 0:
#		print('!!!!!!!!No person detected!!!!')
		return
	iou_scores = []
	for i in range(len(box)):
		iou_scores.append(get_iou(box[i],[x,y,x1,y1]))

	index = np.argmax(iou_scores)
	print("get_coordinates : ", iou_scores, ' ',box, ' ', x, y, x1, y1)

	if np.sum(iou_scores) == 0:
		# print('#'*20, 'No Match found', '#'*20)
		box = np.array(box)
		distance = np.power(((x+x1)/2 - np.array(box[:,0] + box[:,2])/2),2) + np.power(((y+y1)/2 - (box[:,1]+box[:,3])/2), 2)
		index = np.argmin(distance)

	x, y, w, h = box[index][0], box[index][1], (box[index][2]-box[index][0]), (box[index][3]-box[index][1])
	initBB = (x+w//2-60,y+h//2-45,130,130)
    #initBB = (x+w//2-50,y+h//2-50,100,100)  # default  iou box
    

	return initBB, (x,y,x+w,y+h), iou_scores

def get_interArea(boxA, track_boxB):
	xA = max(boxA[0], track_boxB[0])
	yA = max(boxA[1], track_boxB[1])
	xB = min(boxA[2], track_boxB[2])
	yB = min(boxA[3], track_boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	BoxBArea = (track_boxB[2] - track_boxB[0] + 1) * (track_boxB[3] - track_boxB[1] + 1)


	iou = interArea / float(boxAArea + BoxBArea - interArea + 1e-5)

	# return the intersection over union value
	return iou



def get_move(W,track_w, H=0, track_H=0):
    twist = Twist()
    x_bias = 0.1
    z_bias = 0.2
    angular_z = round(-1.0 * (W-track_w)/W,5  ) 
    angular_chk = 1  #  0 : left  ,   1  : right
    if angular_z < 0 :  # 우회전 해야함 
        angular_chk = -1
    
    #if angular_z < 0: 
    
    angular_z += (z_bias*angular_chk)
    # 최소값 정의 angular
    if abs(angular_z) <= 0.37+ abs(z_bias) :
        angular_z = 0.0


    # 최대 값 정의 angular 
    if abs(angular_z) > 0.8 :
        angular_z = 0.8 * angular_chk
    #elif angular_z < -0.8:
    #    angular_z = -0.8

    

    # angular에 값이 들어가 있는 경우
    # angular에 절반 만큼의 속도를 부여
    linear_x = round(abs(angular_z/2) , 5) 
    if  abs(angular_z) > 0: 
        twist.linear.x = linear_x + x_bias
    else : 
        twist.linear.x = 0.25 + x_bias 
    print("angular_z : ", angular_z/2) # cam 상에서 내 위치 : - right   + left   실제 적용해야하는 값 : +  right,    - left  (화면이 반대이므로)
    print("linear_x : ", twist.linear.x) # - right   + left서
    
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = round(angular_z/2 , 5)

    return twist, angular_z


def init_twist():
    twist = Twist()
    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0
    return twist

def time_wait_pub(duration, pub,linear=0.2, angular=0.3):
    twist = Twist()
    twist.linear.x = linear; twist.linear.y = 0; twist.linear.z = 0

    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
    time_chk = time.time()
    while time.time() - time_chk < duration : 
        pub.publish(twist)

def time_wait(duration):
    time_chk = time.time()
    #while time.time() - time_chk < duration :
    time.sleep(duration)


# =============================== hand option에 따라 움직임을 재어할때 사용 ===============================
def hand_option(action,pub) :
    twist = Twist()

    if action == 'move':
        twist.linear.x = 0.3; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0

    elif action == "stop" :
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        
    pub.publish(twist)

def main(video_path,pub):
    # Definition of the parameters
    init_first = True
    
    iou_ =0.0
    #algo = "kcf"  # 최소 8 ~ 21 
    #algo = "mosse" 오류 발생
    algo = "csrt" # 오류 발생 
    #algo = "boosting"
    #success = True
    success = False
    #DETECT_AFTER = 100 # 초기 frame 반복당 tracker initialize
    DETECT_AFTER = 50
    frame_number = DETECT_AFTER -2
    x,y,w,h = 170,190,50,130
    
    pre_frame_chk = frame_number
    #cap = cv2.VideoCapture(video_path)
    #_, frame = cap.read()
    dc = DepthCamera() 
    ret, _, frame = dc.get_frame()
    
    init_BB = ( x, y, 100, 100)
    hand_initBB = (0,0,0,0)
    (H, W) = frame.shape[:2]
    tracker = OPENCV_OBJECT_TRACKERS[algo]()
    #tracker.init(frame, init_BB)

    # hand variable
    action = ''
    #hand_gesture = {0:'stop', 5:'move'}
    #knn, angle, mp_hands, mp_drawing, hands = hand_init()

    # ros
    angular_z_pre = 0.0
    angular_z = 0.0
    occlusion = False
    twist = Twist()

    r = rospy.Rate(30)

    while ret:
        frame_number+=1
        ret, depth_frame, frame = dc.get_frame()
        frame = cv2.flip(frame, 1)
        image_np = np.array(frame)
        
        if not ret:
            break
        
        # cam의 화면과 손의 위치가 동일하게 뒤집어줌
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_input = cv2.resize(image_np, (INPUT_SIZE, INPUT_SIZE))
        frame_input = frame_input / 255.
        frame_input = frame_input[np.newaxis, ...].astype(np.float32)
        frame_input = tf.constant(frame_input)
        start_time = time.time()

        # model에 frame input을 넣어서 예측
        pred_bbox = infer(frame_input)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # tf.image.combined_non_max_suppression 
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD
        )

        # 데이터를 numpy 요소로 변환 및 사용하지 않는 요소는 잘라낸다. 
        #print("1. bboxes : ",  boxes)
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        #print("2. bboxes : ",  bboxes)
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        img = frame.copy()


        # ========== person detection end ===================
        # ================ hand detection start ===============
        if init_first and num_objects ==1:

            #  move or stop  동작을 받아야 다음 step 가능

            if action == '' :
                x_min, y_min = bboxes[0][:2]
                action, (hand_x, hand_y, hand_w, hand_h) = hand_detection(dc, x_min, y_min)



        # =============== hand detection end =================
            # tracker 초기화qqqqq
            # get_coordinates(bbox, 해당 bbox 안에 추적을 할 bounding_box)
            # bboxes : x_min, y_min, x_max, y_max  [사람의 bounding box]
            
            #hand_initBB, trueBB,iou_ = get_coordinates(bboxes, min(x_min+20, W), max(y_min-10, 0), (hand_x + hand_w) -20, hand_y+hand_h)
            hand_initBB, trueBB,iou_ = get_coordinates(bboxes, x_min, y_min, (hand_x + hand_w), (hand_y+hand_h) )
            hand_initBB = tuple(map( int, hand_initBB) )
            tracker = OPENCV_OBJECT_TRACKERS[algo]()
            tracker.init(frame, hand_initBB)

            (success, box) = tracker.update(frame)
            if success :
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
                cv2.putText(frame, f"hand tracking", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)
                pred_bbox = [bboxes, scores, classes, num_objects]
                result = utils.draw_bbox(frame, pred_bbox,pub)

            init_first = False
        

        # ================== new hand detection move check ===================


        # 100 frame 단위로 DETECT trackes 확인  DETEVTER
        #if frame_number % DETECT_AFTER == (DETECT_AFTER-1) and num_objects==1 or not success :
        if not init_first and frame_number % DETECT_AFTER == (DETECT_AFTER-1) or not success  :
            if num_objects>=1 :
                #img, yolo_box = yolo_output(frame.copy(),model,['person'], confidence, nms_thesh, CUDA, inp_dim)
                
                #initBB = hand_initBB 

                hand_initBB = tuple(map( int, hand_initBB) )
                boxes = bboxes[0]
                #_, _, hand_iou = get_coordinates([hand_initBB], boxes[0], boxes[1] , boxes[0]+boxes[2], boxes[1]+ boxes[3] )

                # bounding box test _ with hand bbox main
                initBB, trueBB,iou_ = get_coordinates(bboxes, x, y, x+w, y+h)
                #initBB, trueBB,iou_ = get_coordinates([hand_initBB], x, y, x+w, y+h)
                #initBB = hand_initBB
                initBB = tuple(map( int, initBB) )
                trueBB = tuple(map( int, trueBB) )

                cv2.putText(img, f"iou_scores : {iou_} "  ,  (initBB[0] , initBB[1] )  , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

                cv2.rectangle(img, (initBB[0], initBB[1]), (initBB[0] + initBB[2], initBB[1] + initBB[3]),(255, 255, 0), 2)
                cv2.putText(img, f"old initBB box : {initBB} ",  ( (initBB[0] + initBB[2] - 30) , ( initBB[1] + initBB[3]+20 ) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (102, 160, 145), 2)

                cv2.rectangle(img, (trueBB[0], trueBB[1]), (trueBB[0] + trueBB[2], trueBB[1] + trueBB[3]),(255, 255, 0), 2)
                cv2.putText(img, f"trueBB : {iou_} ",  (trueBB[2], trueBB[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

                
                #initBB =  ( trueBB[0] + (trueBB[2]//2) - (trueBB[2]//4) + (init_BB[2]//2) ) , ( trueBB[1] + (trueBB[3]//2) - (trueBB[3]//4) + (init_BB[3]//2)  )  , initBB[2], initBB[3]  
                #initBB = tuple(map( lambda x: max(0,int(x) ) , initBB) )

                #print(f"new initBB :", initBB)

                #cv2.rectangle(img, (initBB[0], initBB[1]), (initBB[0] + initBB[2], initBB[1] + initBB[3]),(147, 200, 100), 2)
                #cv2.putText(img, f"new initBB box : {initBB} ",  ( (initBB[0] + initBB[2] - 30) , ( initBB[1] + initBB[3]+20 ) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (102, 160, 145), 2)
                
                cv2.imshow('yolo', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                tracker = OPENCV_OBJECT_TRACKERS[algo]()

                print("initBB : ", initBB)
                tracker.init(frame, initBB)

        
        (success, box) = tracker.update(frame)
        print("box : ", box)
        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),(0, 255, 0), 2)
        #pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        #pre_frame_chk = DETECT_AFTER-1
        if not init_first and  success :
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 120), 2)
            
           # distance 1m내인지 여부 판단 
            w_coverage = [0]*w
            h_coverage = []
            try: 
                for ih in range(h):
                    for iw in range(w):
                        dist = depth_frame.get_distance(x+iw, y+ih) 
                        if 0.7 < dist and dist < 2.5:
                            w_coverage[iw] += 1
                    h_coverage.append(sum(w_coverage) > 0)                  
                    w_coverage = [0]*w
                ok = sum(h_coverage) > 0
            except:
                ok = False
     
            cv2.putText(frame, f"current : {ok} ", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)
            #cv2.putText(frame, f"current tracking", (x + w//2, y+h +10), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 120, 120), 2)

            pred_bbox = [bboxes, scores, classes, num_objects]
            result = utils.draw_bbox(frame, pred_bbox)

            twist,angular_z = get_move(W//2, x+(w//2) )
            #r.sleep()
        #else :
            
        if not init_first :
            if num_objects >=1 :
                pre_frame_chk = frame_number

            elif num_objects <1 or not ok:
                occlusion = True
                twist.linear.x =0
                twist.angular.z =0
                pub.publish(twist)
                # if  frame_number - pre_frame_chk > 15 :
                #     time.sleep(1)
                #     dc.release()
                #     cv2.destroyAllWindows()
                    
                #     break
                
                '''
                if  frame_number - pre_frame_chk > 5 :
                    
                    twist.linear.x =0
                    twist.angular.z =0
                    pub.publish(twist)
                    #time_stop = time.time()
                    # check=True
                    # 1초간 대기
                    cv2.putText(frame, f"num_objects <1  time.sleep(1.2)", (W//2-50, H//2 +80), cv2.FONT_HERSHEY_SIMPLEX,2,(207, 255, 229), 2)
                    time.sleep(1.2)
                '''

            z_bias = 0.05
            if abs(angular_z_pre) - abs(angular_z) > 0.25 + z_bias: 
                print(f"occlusion 발생  :  {occlusion }")
                occlusion = not occlusion
                time_wait_pub(1, pub,linear=0.3, angular=angular_z_pre)

            if not occlusion and ok:
                print(" not occlusion, 정상 publish")
                pub.publish(twist)

            occlusion = False

            angular_z_pre = angular_z

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)


        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            dc.release()
            #cap.release()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':

    rospy.init_node('scout_ros_detector_test', anonymous=False)
    pub = rospy.Publisher('/cmd_vel', 
        Twist, 
        queue_size=10)
    video_path = -1
    main(video_path,pub)
