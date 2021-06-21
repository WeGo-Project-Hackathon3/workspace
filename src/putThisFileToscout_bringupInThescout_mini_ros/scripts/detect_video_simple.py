
import time
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from core.config import cfg
import matplotlib.pyplot as plt

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

MODEL_PATH = './checkpoints/yolov4-tiny-416'
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

def main(video_path):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)





    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_input = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        frame_input = frame_input / 255.
        frame_input = frame_input[np.newaxis, ...].astype(np.float32)
        frame_input = tf.constant(frame_input)
        start_time = time.time()

        pred_bbox = infer(frame_input)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

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
        print("1. bboxes : ",  boxes)
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        print("2. bboxes : ",  bboxes)
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        print("3. bboxes : ",  bboxes)
        #pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        pred_bbox = [bboxes, scores, classes, num_objects]
        
        '''
        # ====================== YOLO 지정 class만 detection ==================
        # YOLO가 가지는 모든 class name 가지고 온다 80개 
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # only person detection
        allowed_classes = ['person']

        ###

        # 반복을 돌려서 allowed_classes 목록의 클래스 만 허용
        #''
        names,deleted_index =[], []

        for i in range(num_objects) :
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name not in allowed_classes :
                # allowed에 포함되지 않은 class 찾기
                deleted_index.append(i)
            else : 
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        
        cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        print("Objects being tracked: {}".format(count))
        print("전 len(bboxes) : ", len(bboxes))
        print("전 bboxes : ", bboxes)
        # allowed_classes 없는 class 삭제
        bboxes = np.delete(bboxes, deleted_index, axis=0)
        scores = np.delete(scores, deleted_index, axis=0)

        print("후 len(bboxes) : ", len(bboxes))
        print("후 bboxes : ", bboxes)
        #
        #pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #if len(bboxes) >0 :
        #    pred_bbox = [bboxes, scores, classes, num_objects]
        
        #
        '''
        '''
        # ================ deep-sort====================
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        '''
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)


        #if len(bboxes) >0 :
        result = utils.draw_bbox(frame, pred_bbox)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #video_path = './data/road.mp4'
    video_path = -1
    main(video_path)
