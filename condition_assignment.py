import cv2
import json

import numpy as np
import pandas as pd
import scipy.signal
import scipy.spatial

import face_recognition


def get_ear(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = scipy.spatial.distance.euclidean(eye[1], eye[5])
    B = scipy.spatial.distance.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = scipy.spatial.distance.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear


def processFrames(data_folder):
    result = list()
    video = cv2.VideoCapture(f'{data_folder}/world.mp4')
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in range(video_length):
        d = dict()
        d['frame'] = frame_number

        if frame_number % 100 == 0:
            print(f'Processing {frame_number}/{video_length} frames.')

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = video.read()

        face_landmarks = face_recognition.face_landmarks(image)

        if len(face_landmarks) > 0:
            face_landmarks = face_landmarks[0]
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']

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

        cv2.imwrite(f'frames/{frame_number}.jpg', image)
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
    data_folder = '2021_01_24/001'
    
    result = processFrames(data_folder)
    result = filterNoise(result, kernel_size=15)  # only interested in longer periods of closed eyes
    df_result = pd.DataFrame(result)
    df_result.to_excel('conditions.xlsx', index=False)
    


if __name__ == '__main__':
    main()