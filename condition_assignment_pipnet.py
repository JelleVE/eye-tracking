import cv2
import json

import numpy as np
import pandas as pd
import scipy.signal
import scipy.spatial

# import face_recognition
from PIPNet.pipnet import PIPNet


def getStartAndEndFrame(video_path, starting_hours, starting_minutes, starting_seconds, ending_hours, ending_minutes, ending_seconds):
    video = cv2.VideoCapture(video_path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)

    frame_start = int((starting_hours*60*60 + starting_minutes*60 + starting_seconds) * video_fps)
    frame_end = int((ending_hours*60*60 + ending_minutes*60 + ending_seconds) * video_fps)

    frame_start = max(0, frame_start - 10*video_fps)
    frame_end = min(video_length, frame_end + 10*video_fps)

    return frame_start, frame_end


def get_ear(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = scipy.spatial.distance.euclidean(eye[1], eye[7])
    B = scipy.spatial.distance.euclidean(eye[2], eye[6])
    C = scipy.spatial.distance.euclidean(eye[3], eye[5])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    D = scipy.spatial.distance.euclidean(eye[0], eye[4])
 
    # compute the eye aspect ratio
    ear = (A + B + C) / (3.0 * D)
 
    # return the eye aspect ratio
    return ear


def processFrames(video_path, pipnet, frame_start, frame_end):
    result = list()
    video = cv2.VideoCapture(video_path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in range(video_length):
        if frame_number < frame_start or frame_number > frame_end: # don't process frames out of specified range
            continue

        d = dict()
        d['frame'] = frame_number

        if frame_number % 100 == 0:
            print(f'Processing {frame_number}/{video_length} frames.')

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = video.read()

        # face_landmarks = face_recognition.face_landmarks(image)

        face_detections = pipnet.detectFaces(image)
        assert len(face_detections) == 0 or len(face_detections) == 1

        if len(face_detections) == 1:
            face_landmarks = pipnet.detectLandmarks(image, face_detections[0])

        if len(face_detections) > 0:
            left_eye = face_landmarks['eye_left']
            right_eye = face_landmarks['eye_right']

            ear_left = get_ear(left_eye)
            ear_right = get_ear(right_eye)
            
            closed = ear_left < 0.22 and ear_right < 0.22

            if closed:
                d['face_present'] = True
                d['condition_raw'] = 'eyes closed'
            else:
                d['face_present'] = True
                d['condition_raw'] = 'eyes open'
        else:
            d['face_present'] = False
            d['condition_raw'] = 'eyes open'

        # cv2.imwrite(f'frames/{frame_number}.jpg', image)
        result.append(d)

    return result


def filterNoise(l, kernel_size):
    """
    Map eyes closed to 1, eyes open & no face to 0.
    Use a median filter to do (initial?) filtering.
    """
    mapping = {
        'eyes closed': 1,
        'eyes open': 0,
    }
    inv_map = {v: k for k, v in mapping.items()}

    signal = np.array([mapping[el['condition_raw']] for el in l], int)
    signal_processed = scipy.signal.medfilt(signal, kernel_size=kernel_size)

    result = list()
    for el, sig_proc in zip(l, signal_processed):
        el['condition_processed'] = inv_map[sig_proc]
        result.append(el)
    return result



def main():
    pipnet = PIPNet()
    data_folder = '../eye-tracking-data/2021_01_24/001'
    video_path = f'{data_folder}/world.mp4'
    
    result = processFrames(video_path, pipnet)
    result = filterNoise(result, kernel_size=15)  # only interested in longer periods of closed eyes
    df_result = pd.DataFrame(result)
    df_result.to_excel('conditions_pipnet.xlsx', index=False)
    


if __name__ == '__main__':
    main()