import os
import json
import shapely
import shapely.geometry

import multiprocessing

from PIL import Image, ImageDraw
import face_recognition



def createBoundingBox(points):
    mp = shapely.geometry.MultiPoint(points)
    minx, miny, maxx, maxy = mp.bounds
    return [(minx, miny), (maxx, maxy)]



def processFace(fn):
    """
    Only for testing purposes
    """
    try:
        image = face_recognition.load_image_file(f'frames/{fn}')
        pil_image = Image.fromarray(image)
        # draw = ImageDraw.Draw(pil_image)
        
        # image_height = image.shape[0]
        # image_width = image.shape[1]

        # Draw gaze
        # pos_x = int(round(image_width * current_datum['norm_pos'][0]))
        # pos_y = int(round(image_height * (1-current_datum['norm_pos'][1])))
        # r = 15
        # draw.ellipse((pos_x-r, pos_y-r, pos_x+r, pos_y+r), width=3)
        
        # Draw landmarks
        face_landmarks = face_recognition.face_landmarks(image)
        if len(face_landmarks) > 0:
            pil_image.save(f'faces/yes/{fn}')
        else:
            pil_image.save(f'faces/no/{fn}')

        # points_eyes = face_landmarks[0]['left_eyebrow'] + face_landmarks[0]['right_eyebrow'] + face_landmarks[0]['left_eye'] + face_landmarks[0]['right_eye']
        # points_nose = face_landmarks[0]['nose_tip'] + face_landmarks[0]['nose_bridge']
        # points_mouth = face_landmarks[0]['top_lip'] + face_landmarks[0]['bottom_lip']

        # rect_eyes = createBoundingBox(points_eyes)
        # rect_nose = createBoundingBox(points_nose)
        # rect_mouth = createBoundingBox(points_mouth)

        # draw.rectangle(rect_eyes)
        # draw.rectangle(rect_nose)
        # draw.rectangle(rect_mouth)

        # print(face_landmarks_list[0].keys())
        # for face_landmarks in face_landmarks_list:
        #     for facial_feature in face_landmarks.keys():
        #         draw.line(face_landmarks[facial_feature], width=5)

        # fn_out = f'{str(current_datum["timestamp_gaze"]).replace(".", "_")}.png'
        # pil_image.save(f'processed/{fn_out}')
    except IndexError:
        pass



def processGaze(current_datum):
    """
    Only for testing purposes
    """
    try:
        image = face_recognition.load_image_file(f'frames/{current_datum["frame_number"]}.png')
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Draw gaze
        pos_x = int(round(image_width * current_datum['norm_pos'][0]))
        pos_y = int(round(image_height * (1-current_datum['norm_pos'][1])))
        r = 15
        draw.ellipse((pos_x-r, pos_y-r, pos_x+r, pos_y+r), width=3)
        
        # Draw landmarks
        # face_landmarks = face_recognition.face_landmarks(image)
        # if len(face_landmarks) > 0:
        #     pil_image.save(f'faces/yes/{fn}')
        # else:
        #     pil_image.save(f'faces/no/{fn}')

        # points_eyes = face_landmarks[0]['left_eyebrow'] + face_landmarks[0]['right_eyebrow'] + face_landmarks[0]['left_eye'] + face_landmarks[0]['right_eye']
        # points_nose = face_landmarks[0]['nose_tip'] + face_landmarks[0]['nose_bridge']
        # points_mouth = face_landmarks[0]['top_lip'] + face_landmarks[0]['bottom_lip']

        # rect_eyes = createBoundingBox(points_eyes)
        # rect_nose = createBoundingBox(points_nose)
        # rect_mouth = createBoundingBox(points_mouth)

        # draw.rectangle(rect_eyes)
        # draw.rectangle(rect_nose)
        # draw.rectangle(rect_mouth)

        # print(face_landmarks_list[0].keys())
        # for face_landmarks in face_landmarks_list:
        #     for facial_feature in face_landmarks.keys():
        #         draw.line(face_landmarks[facial_feature], width=5)

        fn_out = f'{str(current_datum["timestamp_gaze"]).replace(".", "_")}.png'
        pil_image.save(f'processed/{fn_out}')
    except IndexError:
        pass



def process(current_datum):
    try:
        image = face_recognition.load_image_file(f'frames/{current_datum["frame_number"]}.png')
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Draw gaze
        pos_x = int(round(image_width * current_datum['norm_pos'][0]))
        pos_y = int(round(image_height * (1-current_datum['norm_pos'][1])))
        r = 15
        draw.ellipse((pos_x-r, pos_y-r, pos_x+r, pos_y+r), width=3)
        
        # Draw landmarks
        face_landmarks = face_recognition.face_landmarks(image)
        points_eyes = face_landmarks[0]['left_eyebrow'] + face_landmarks[0]['right_eyebrow'] + face_landmarks[0]['left_eye'] + face_landmarks[0]['right_eye']
        points_nose = face_landmarks[0]['nose_tip'] + face_landmarks[0]['nose_bridge']
        points_mouth = face_landmarks[0]['top_lip'] + face_landmarks[0]['bottom_lip']

        rect_eyes = createBoundingBox(points_eyes)
        rect_nose = createBoundingBox(points_nose)
        rect_mouth = createBoundingBox(points_mouth)

        draw.rectangle(rect_eyes)
        draw.rectangle(rect_nose)
        draw.rectangle(rect_mouth)

        # print(face_landmarks_list[0].keys())
        # for face_landmarks in face_landmarks_list:
        #     for facial_feature in face_landmarks.keys():
        #         draw.line(face_landmarks[facial_feature], width=5)

        fn_out = f'{str(current_datum["timestamp_gaze"]).replace(".", "_")}.png'
        pil_image.save(f'processed/{fn_out}')
    except IndexError:
        pass



def main():
    fns = os.listdir('frames/')

    pool = multiprocessing.Pool()
    pool.map(processFace, fns)
    pool.close()

    # with open('preprocessed.json', 'r') as f:
    #     ts_data = json.load(f)

    # pool = multiprocessing.Pool()
    # pool.map(processGaze, ts_data)
    # pool.close()
        



if __name__ == '__main__':
    main()