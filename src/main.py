import os
import sys
import cv2
import logging
import time
from argparse import ArgumentParser
from openvino.inference_engine import IECore

from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarks
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from visualize import Visualizer
import pyautogui
import datetime

log = logging.getLogger(__name__)

model_precision_fd = 'FP32'

# model_precision = 'FP32'
# model_precision = 'FP16-INT8'
model_precision = 'FP32'

# Default args
m_fd = '../intel/face-detection-0200/' + model_precision_fd +'/face-detection-0200'
m_hpe = '../intel/head-pose-estimation-adas-0001/' + model_precision + '/head-pose-estimation-adas-0001'
m_ld = '../intel/landmarks-regression-retail-0009/' + model_precision + '/landmarks-regression-retail-0009'
m_ge = '../intel/gaze-estimation-adas-0002/' + model_precision + '/gaze-estimation-adas-0002'
input_stream = '../bin/demo.mp4'

def build_argparser():
    '''
    Parse command line arguments.
    :return: command line arguments
    '''
    parser = ArgumentParser(description = 'Eye Gaze Computer Pointer Controller')
    required = parser.add_argument_group('required arguments')

    required.add_argument('-m_fd', type=str, required=False, default=m_fd, help='Path to a IR model for face detection')
    required.add_argument('-m_hpe', type=str, required=False, default=m_hpe, help='Path to IR model for head pose estimation')
    required.add_argument('-m_ld', type=str, required=False, default=m_ld, help='Path to IR model for facial landmark detection')
    required.add_argument('-m_ge', type=str, required=False, default=m_ge, help='Path to IR model for gaze estimation')
    required.add_argument('-i', type=str, required=False, default=input_stream, help='Path to image or video file, or \'cam\' for live feed')
    required.add_argument('-d', type=str, required=False, default='CPU', help='Target device to infer on: CPU, GPU, FPGA or MYRIAD. (CPU by default)')
    required.add_argument('-pt', type=float, required=False, default=0.65, help='Probablity threshold for face detection, set a number between 0-1')
    required.add_argument('-v', type=str2bool, required=False, default='true', help='Visualization available flag - set to no display by default, to set display frames, specify \'t\' or \'yes\' or \'true\' or \'1\'')
    parser.add_argument('-cpu_ext', type=str, required=False, default=None, help='MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.')
#    parser.add_argument('-flags', '--visualization_flags', type=str, required=False, nargs='+',
    parser.add_argument('-flags', type=str, required=False, nargs='+',
                        default=['fd', 'hpe', 'ld' ,'ge' ],
                        help="Visualization detail flags - set to display different model outputs of each frame. Example: -flag fd hpe ge fld (Seperate each flag by space)"
                             "fd for Face Detection Model, hpe for Head Pose Estimation Model,"
                             "fld for Facial Landmark Detection Model, ge for Gaze Estimation Model.")
    return parser
def str2bool(v):
    #returns True or False
    return v.lower() in ('yes', 'true', 't', '1')

def pipeline(args):
    input_file = args.i
    if input_file.lower()=="cam":
        feed = InputFeeder("cam")
    elif not os.path.isfile(input_file):
            logging.error("Given input file is not found.")
            exit(1)
    else:
        feed = InputFeeder(input_file)
    feed.load_data()

    FaceDetectionPipe = FaceDetection(args.m_fd, args.pt, args.d, args.cpu_ext)
    load_time = time.time()
    FaceDetectionPipe.load_model()
    load_time_fd = time.time() - load_time

    FacialLandmarksPipe = FacialLandmarks(args.m_ld, args.d, args.cpu_ext)
    load_time = time.time()
    FacialLandmarksPipe.load_model()
    load_time_ld = time.time() - load_time

    HeadPoseEstimationPipe = HeadPoseEstimation(args.m_hpe, args.d, args.cpu_ext)
    load_time = time.time()
    HeadPoseEstimationPipe.load_model()
    load_time_hpe = time.time() - load_time

    GazeEstimationPipe = GazeEstimation(args.m_ge, args.d, args.cpu_ext)
    load_time = time.time()
    GazeEstimationPipe.load_model()
    load_time_ge = time.time() - load_time

    log.info('Load time for face detection model: ' + str(load_time_fd))
    log.info('Load time for landmark detection model: ' + str(load_time_ld))
    log.info('Load time for head pose estimation model: ' + str(load_time_hpe))
    log.info('Load time for gaze estimation model: ' + str(load_time_ge))
    # Prepare to save demo_result.mp4
    frame_rate = 10.0
    size = (1000, 500) # 
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # file format is mp4
    writer = cv2.VideoWriter('../bin/demo_result.mp4', fmt, frame_rate, size) # 
    # Prepare for variables
    inf_time_fd = inf_time_ld = inf_time_hpe = inf_time_ge = frame_count = 0
    sum_inf_time_fd = sum_inf_time_ld = sum_inf_time_hpe = sum_inf_time_ge =0
    for frame in feed.next_batch():
        if frame is None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

        frame_count += 1
        inf_time = time.time()
        fd_img_output, fd_coords = FaceDetectionPipe.predict(frame)
        inf_time_fd = time.time() - inf_time

        if (fd_coords == []):
            log.info('No face detected')
        else:
            inf_time = time.time()
            eye_l_image, eye_r_image, ld_coords = FacialLandmarksPipe.predict(fd_img_output.copy())
            inf_time_ld = time.time() - inf_time

            inf_time = time.time()
            hpe_output = HeadPoseEstimationPipe.predict(fd_img_output.copy())
            inf_time_hpe = time.time() - inf_time

            yaw, pitch, roll = hpe_output
            inf_time = time.time()
            mouse_xy, gaze_val = GazeEstimationPipe.predict(eye_l_image, eye_r_image, [yaw, pitch, roll])
            inf_time_ge = time.time() - inf_time

            if frame_count % 5 == 0:
                pointer = MouseController('medium', 'fast')
                pointer.move(mouse_xy[0], mouse_xy[1])
            sum_inf_time_fd = sum_inf_time_fd + inf_time_fd
            sum_inf_time_ld = sum_inf_time_ld + inf_time_ld
            sum_inf_time_hpe = sum_inf_time_hpe + inf_time_hpe
            sum_inf_time_ge = sum_inf_time_ge + inf_time_ge
        
            if (args.v):
                v = Visualizer(frame, fd_img_output, fd_coords, ld_coords, eye_l_image,eye_r_image, hpe_output, mouse_xy,gaze_val)
                vis_frame = v.visualizer(args.flags)
                writer.write(vis_frame)
    # Release the video writer
    writer.release()

    # Calulate average inference time
    avg_inf_time_fd = sum_inf_time_fd / frame_count
    avg_inf_time_ld = sum_inf_time_ld / frame_count
    avg_inf_time_hpe = sum_inf_time_hpe / frame_count
    avg_inf_time_ge = sum_inf_time_ge / frame_count
    # Calculate frames per second
    fps_fd = round(1 / avg_inf_time_fd)
    fps_ld = round(1 / avg_inf_time_ld)
    fps_hpe = round(1 / avg_inf_time_hpe)
    fps_ge = round(1 / avg_inf_time_ge)

    log.info('Save metrics to file...')
    fd_list = [args.m_fd, model_precision_fd, avg_inf_time_fd, fps_fd, load_time_fd]
    ld_list = [args.m_ld, model_precision, avg_inf_time_ld, fps_ld, load_time_ld]
    hpe_list = [args.m_hpe, model_precision, avg_inf_time_hpe, fps_hpe, load_time_hpe]
    ge_list = [args.m_ge, model_precision, avg_inf_time_ge, fps_ge, load_time_ge]
    stats_lists = [fd_list, ld_list, hpe_list, ge_list]
    dt_now = datetime.datetime.now()
    file_name = dt_now.strftime('%Y%m%d_%H%M%S') + 'stats.txt'
    save_metrics_stats(file_name, stats_lists)
    feed.close()

def save_metrics_stats(text_file, stats_lists):
    # If no folder exists, make folders first
    folder_path = '../metrics_stats_result/'
    if os.path.isdir(folder_path)==False:
        os.makedirs(folder_path)
    # Save to file
    with open(os.path.join(folder_path, text_file), 'w') as f:
        dt_now = datetime.datetime.now()
        f.write("Metrics stats of the program run at {}\n".format(dt_now))
        f.write("Information of model name, model_precision, inference time(s), frame per second, load time(s)\n")
        message = []
        for model_v in stats_lists:
            f.write('\n')
            for v in model_v:
                f.write(str(v) + '\n')

def main():
    logging.basicConfig(filename='app.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    args = build_argparser().parse_args()
    pipeline(args)

if __name__ == '__main__':
    main()