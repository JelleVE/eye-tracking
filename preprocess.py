import cv2
import json

import numpy as np
import file_methods


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def processFrames(data_folder, frame_numbers):
    video = cv2.VideoCapture(f'{data_folder}world.mp4')
    for i, frame_number in enumerate(frame_numbers):
        if i % 100 == 0:
            print(f'Processed {i}/{len(frame_numbers)} frames.')

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = video.read()
        cv2.imwrite(f'frames/{frame_number}.png', image)



def retrieveGazeTimestampsAndPositions(data_folder, min_confidence):
    pl_gaze = file_methods.load_pldata_file(data_folder, 'gaze')

    result = list()
    for datum in pl_gaze.data:
        if datum['confidence'] >= min_confidence and datum['norm_pos'][0] >= 0 and datum['norm_pos'][0] <= 1 and datum['norm_pos'][1] >= 0 and datum['norm_pos'][1] <= 1:
            result.append({
                    'timestamp_gaze': datum['timestamp'],
                    'norm_pos': datum['norm_pos']
                })

    return result



def correlateGazeWorld(data_folder, gaze_tsp):
    world_ts = np.load(f'{data_folder}world_timestamps.npy')
    for el in gaze_tsp:
        gaze_ts = el['timestamp_gaze']
        frame_number = np.argmin(np.abs(world_ts - gaze_ts))
        el['frame_number'] = frame_number

    return gaze_tsp



def main():
    data_folder = '2021_02_06/000/'
    gaze_tsp = retrieveGazeTimestampsAndPositions(data_folder, min_confidence=0.9)
    gaze_corr = correlateGazeWorld(data_folder, gaze_tsp)
    
    frame_numbers = [el['frame_number'] for el in gaze_corr]
    processFrames(data_folder, list(set(frame_numbers)))

    with open('preprocessed.json', 'w') as f:
        json.dump(gaze_corr, f, cls=NpEncoder, indent=2)



if __name__ == '__main__':
    main()