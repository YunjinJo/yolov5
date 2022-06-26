# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def checkXYcord(min, max, object):
    if min <= object < max:
        return True
    else:
        return False

# Function drawing traffic lanes
def drawing_box(image_data, traffic_lane, line_col):
    point_LUp = traffic_lane[0]
    point_RUp = traffic_lane[1]
    point_LDo = traffic_lane[2]
    point_RDo = traffic_lane[3]
    line1poly = np.array([point_LUp, point_RUp, point_RDo, point_LDo])
    line1poly = line1poly.reshape(-1, 1, 2)
    return cv2.polylines(image_data, [line1poly], True, line_col)

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        draw_boxes=[[[50, 50], [100, 50], [50, 100], [100, 100]], \
                    [[50, 50], [100, 50], [50, 100], [100, 100]], \
                    [[50, 50], [100, 50], [50, 100], [100, 100]]], # use to draw traffic lanes
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            '''
            # 이 코드는 상관없음
            # 차량 카운팅
            class_name_count = 'cars' # 어떤 클래스를 카운팅 할건지
            l = s[1:s.find(class_name_count)].split()[-1]
            if class_name_count in s:
                print(l, class_name_count) # 출력할 글자 (차량 대수, 클래스 이름)
                cv2.rectangle(im0, (0,0), (1100,250), (0,0,0), -1) # 글자 배경 (삭제해도 무방)
                cv2.putText(im0, l + class_name_count, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,255,255), 25, cv2.LINE_AA) # 글자 크기, 폰트, 색상
            '''
            '''
            # 해상도 비율 조정
            original_h = 3000
            original_w = 4000

            changed_h = 1080
            changed_w = 1920

            ratio_h = round(original_h/changed_h)
            ratio_w = round(original_w/changed_w)
            '''

            '''
            # 각 차선별로 구역 지정 동영상의 경우 해상도 1920x1080
            line1xmin = 892
            line1ymin = 493
            line1xmax = 1009
            line1ymax = 1073

            line2xmin = 1001
            line2ymin = 483
            line2xmax = 1257
            line2ymax = 1077

            line3xmin = 1112
            line3ymin = 487
            line3xmax = 1497
            line3ymax = 1075
            '''

            str_b = draw_boxes # 터미널을 통해 입력한 좌표 배열이 문자열로 입력됨
            traffic_lanes = 3
            points = 4
            num_coor = 2

            temp_str_num = [] # x 또는 y 좌표
            temp_list_coor = [] # (x, y) 순서쌍
            result_coor = np.zeros((traffic_lanes, points, num_coor), dtype=int) # 최종 좌표 배열
            row_count = 0 # 행
            col_count = 0 # 열
            # 문자열로 입력된 좌표 배열을 좌표 행렬로 변환; 차선을 그리기 위해 필요한 좌표 행렬
            for n in range(len(str_b)):
                # 분리된 x, y 좌표 문자를 int로 형 변환; '8', '0', '2' -> 8, 0, 2 과정
                if str_b[n] != '[' and str_b[n] != ']' and str_b[n] != ',':
                    if type(int(str_b[n])) == int:
                        temp_str_num.append(int(str_b[n]))
                # x, y 좌표 숫자를 x, y 좌표 값으로 변환
                elif str_b[n] == ',':
                    if len(temp_str_num) == 2:
                        temp_list_coor.append(temp_str_num[0] * 10 + \
                                              temp_str_num[1] * 1)
                    elif len(temp_str_num) == 3:
                        temp_list_coor.append(temp_str_num[0] * 100 + \
                                              temp_str_num[1] * 10 + \
                                              temp_str_num[2] * 1)
                    elif len(temp_str_num) == 4:
                        temp_list_coor.append(temp_str_num[0] * 1000 + \
                                              temp_str_num[1] * 100 + \
                                              temp_str_num[2] * 10 + \
                                              temp_str_num[3] * 1)
                    temp_str_num = [] # 숫자 초기화
                    # x, y 좌표 값을 (x, y) 순서쌍으로 변환
                    if str_b[n - 1] == ']':
                        result_coor[row_count][col_count] = temp_list_coor
                        col_count = col_count + 1 # 열 이동
                        temp_list_coor = [] # 순서쌍 초기화
                # 열 초기화 및 행 이동; for문 중첩을 피하기 위함 & 시간 복잡도 최소화
                elif n > 2 and str_b[n] == '[' and str_b[n - 1] == '[':
                    col_count = 0
                    row_count = row_count + 1
                # 앞선 과정으로 변환되지 않은 마지막 x, y 좌표 값을 (x, y) 순서쌍으로 변환
                elif str_b[n] == ']' and str_b[n - 1] == ']' and str_b[n - 2] == ']':
                    if len(temp_str_num) == 2:
                        temp_list_coor.append(temp_str_num[0] * 10 + \
                                              temp_str_num[1] * 1)
                    elif len(temp_str_num) == 3:
                        temp_list_coor.append(temp_str_num[0] * 100 + \
                                              temp_str_num[1] * 10 + \
                                              temp_str_num[2] * 1)
                    elif len(temp_str_num) == 4:
                        temp_list_coor.append(temp_str_num[0] * 1000 + \
                                              temp_str_num[1] * 100 + \
                                              temp_str_num[2] * 10 + \
                                              temp_str_num[3] * 1)
                    # x, y 좌표 값을 (x, y) 순서쌍으로 변환
                    if str_b[n - 1] == ']':
                        result_coor[row_count][col_count] = temp_list_coor # 좌표 행렬 초기화
                        col_count = col_count + 1
                        temp_list_coor = []

            result_coor = result_coor.tolist() # 문자열에서 변환된 좌표 행렬을 numpy에서 배열로 형 변환

            # result_coor[n차선, n = 1, 2, 3][좌상, 우상, 좌하, 우하][x, y]
            line1xmin = result_coor[0][0][0]
            line1ymin = result_coor[0][0][1]
            line1xmax = result_coor[0][1][0]
            line1ymax = result_coor[0][2][1]

            line2xmin = result_coor[1][0][0]
            line2ymin = result_coor[1][0][1]
            line2xmax = result_coor[1][1][0]
            line2ymax = result_coor[1][2][1]

            line3xmin = result_coor[2][0][0]
            line3ymin = result_coor[2][0][1]
            line3xmax = result_coor[2][1][0]
            line3ymax = result_coor[2][2][1]

            # 차선별로 상자 그리기; 1차선, 2차선, 3차선
            drawing_box(im0, result_coor[0], (255, 0, 0))
            drawing_box(im0, result_coor[1], (0, 255, 0))
            drawing_box(im0, result_coor[2], (0, 0, 255))

            #print('\n',(pred[0][3][5]), len(pred[0]))  # 테스트용

            # 각 라인별 몇 대의 차량이 있는지 세기 위한 변수
            line1 = 0
            line2 = 0
            line3 = 0

            # 라인별 차량 카운트
            for list_num in range(0, len(pred[0]) - 1):
                if pred[0][list_num][5] == 2. or pred[0][list_num][5] == 7. or pred[0][list_num][5] == 3. : # 2. -> 일반 차량, 7. -> 트럭, 3. -> 오토바이
                    xmin = pred[0][list_num][0] # 탐지된 차량 박스의 xmin 좌표
                    #print('\n',xmin)
                    ymin = pred[0][list_num][1] # 탐지된 차량 박스의 ymin 좌표
                    #print('\n',ymin)
                    xmax = pred[0][list_num][2]
                    ymax = pred[0][list_num][3]
                    xcord = (xmin + xmax) / 2   # 박스의 중앙값 계산
                    ycord = (ymin + ymax) / 2
                    cv2.circle(im0, (math.floor(xcord), math.floor(ycord)), 10, (0, 0, 255), -1) # 박스 중간에 원을 그림
                    # 라인별 차량 수 카운트
                    if checkXYcord(line1xmin, line1xmax, xcord) and (line1ymin <= ycord < line1ymax):
                        line1 += 1
                    elif (line2xmin <= xcord < line2xmax) and (line2ymin <= ycord < line2ymax):
                        line2 += 1
                    elif (line3xmin <= xcord < line3xmax) and (line3ymin <= ycord < line3ymax):
                        line3 += 1

                # 라인별 차량 수 출력
            cv2.putText(im0, 'Line1: ' + str(line1), (line1xmin, line1ymin - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0), 3)
            cv2.putText(im0, 'Line2: ' + str(line2), (line2xmin, line2ymin - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0), 3)
            cv2.putText(im0, 'Line3: ' + str(line3), (line3xmin, line3ymin - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255), 3)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def sum_test(a):
    sum_r = sum(a)
    return sum_r

list_default = [[[50, 50], [100, 50], [50, 100], [100, 100]], \
                [[50, 50], [100, 50], [50, 100], [100, 100]], \
                [[50, 50], [100, 50], [50, 100], [100, 100]]]
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    parser.add_argument('--draw-boxes', type=list, default=list_default, help='draw traffic lanes for counting cars')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)