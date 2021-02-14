import cv2
import os
import json
import shapely
import shapely.ops
import shapely.geometry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing
import collections

from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
# import face_recognition

import face_alignment
from skimage import io


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def processFrame(fixation_id, frame_number, video, norm_pos_x, norm_pos_y):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = video.read()

    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
        input = io.imread('tmp3.jpg')
        preds = fa.get_landmarks(input)[-1]

        plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

        pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
        pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                      'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                      'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                      'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                      'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                      'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                      'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                      'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                      'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                      }

        print(pred_types)

        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input)

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color, **plot_style)

        ax.axis('off')
        plt.show()

        
        # pil_image = Image.fromarray(image)
        # draw = ImageDraw.Draw(pil_image)
        
        # image_height = image.shape[0]
        # image_width = image.shape[1]

        # # Draw gaze
        # pos_x = int(round(image_width * norm_pos_x))
        # pos_y = int(round(image_height * (1-norm_pos_y)))
        # r = 5
        # draw.ellipse((pos_x-r, pos_y-r, pos_x+r, pos_y+r), width=3)
        
        # # Draw landmarks
        # face_landmarks = face_recognition.face_landmarks(image)

        # points_eyes = face_landmarks[0]['left_eyebrow'] + face_landmarks[0]['right_eyebrow'] + face_landmarks[0]['left_eye'] + face_landmarks[0]['right_eye']
        # points_nose = face_landmarks[0]['nose_tip'] + face_landmarks[0]['nose_bridge']
        # points_mouth = face_landmarks[0]['top_lip'] + face_landmarks[0]['bottom_lip']

        # # Define points
        # mean_left_eye = np.mean(np.array(face_landmarks[0]['left_eye']), axis=0)
        # mean_right_eye = np.mean(np.array(face_landmarks[0]['right_eye']), axis=0)
        # mean_nose = np.mean(np.array(face_landmarks[0]['nose_tip']), axis=0)
        # mean_mouth = np.mean(np.array(face_landmarks[0]['top_lip'] + face_landmarks[0]['bottom_lip']), axis=0)
        # mean_points = np.vstack([mean_left_eye, mean_right_eye, mean_nose, mean_mouth])

        # # Voronoi
        # voronoi_polys = list()

        # vor = Voronoi(mean_points)
        # regions, vertices = voronoi_finite_polygons_2d(vor, radius=1e4)

        # min_x = 0
        # max_x = image_width
        # min_y = 0
        # max_y = image_height

        # mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        # bounded_vertices = np.max((vertices, mins), axis=0)
        # maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
        # bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

        # box = shapely.geometry.Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

        # for region in regions:
        #     polygon = vertices[region]
        #     # Clipping polygon
        #     poly = shapely.geometry.Polygon(polygon)
        #     poly = poly.intersection(box)
        #     voronoi_polys.append(poly)
            
        #     plt.fill(*zip(*polygon), alpha=0.4)

        # # Create ROI
        # regions = ['left_eye', 'right_eye', 'nose', 'mouth']
        # mps = [mean_left_eye, mean_right_eye, mean_nose, mean_mouth]
        # roi_mapping = dict()
        # for index, mp in enumerate(mps):
        #     p = shapely.geometry.Point(mp)
        #     for poly in voronoi_polys:
        #         if p.intersects(poly):
        #             roi_mapping[regions[index]] = dict()
        #             roi_mapping[regions[index]]['voronoi_poly'] = poly
        #             roi_mapping[regions[index]]['mp'] = mp
        #             break

        # # Create circles
        # for region in roi_mapping.keys():
        #     relevant_poly = roi_mapping[region]['voronoi_poly']
        #     relevant_mp = roi_mapping[region]['mp']

        #     c = shapely.geometry.Point(relevant_mp).buffer(50)
        #     roi_mapping[region]['intersected_poly'] = c.intersection(relevant_poly)

        # # Draw polys
        # for region in roi_mapping.keys():
        #     poly = roi_mapping[region]['intersected_poly']
        #     coordinates = list(zip(*poly.exterior.coords.xy))
        #     for i in range(len(coordinates)-1):
        #         start = coordinates[i]
        #         stop = coordinates[i+1]
        #         draw.line([start, stop], width=3)

        # p_fixation = shapely.geometry.Point([pos_x, pos_y])

        # bool_eyes = p_fixation.intersects(roi_mapping['left_eye']['intersected_poly']) or p_fixation.intersects(roi_mapping['right_eye']['intersected_poly'])
        # bool_nose = p_fixation.intersects(roi_mapping['nose']['intersected_poly'])
        # bool_mouth = p_fixation.intersects(roi_mapping['mouth']['intersected_poly'])

        # # for facial_feature in face_landmarks[0].keys():
        # #     draw.line(face_landmarks[0][facial_feature], width=5)

        # fn_out = f'processed/{fixation_id}/{frame_number}.png'
        # os.makedirs(os.path.dirname(fn_out), exist_ok=True)
        # pil_image.save(fn_out)

        # return bool_eyes, bool_nose, bool_mouth
    except IndexError:
        pass

    return None


def processFixation(row, video):
    fixation_id = row['id']
    start_frame_index = row['start_frame_index']
    end_frame_index = row['end_frame_index']
    norm_pos_x = row['norm_pos_x']
    norm_pos_y = row['norm_pos_y']

    result_sequence = ['eyes', 'nose', 'mouth']
    result_counts = [0, 0, 0]
    for current_frame_index in range(start_frame_index, end_frame_index+1):
        current_result = processFrame(fixation_id, current_frame_index, video, norm_pos_x, norm_pos_y)
        if current_result is not None:
            bool_eyes, bool_nose, bool_mouth = current_result
            result_counts[0] += bool_eyes
            result_counts[1] += bool_nose
            result_counts[2] += bool_mouth

    index_max = np.argmax(result_counts)
    if result_counts[index_max] == 0:
        return 'none'
    else:
        return result_sequence[index_max]


def main():
    dir_recording = '2021_02_06/001'
    fn_fixations = f'{dir_recording}/exports/000/fixations.csv'
    video = cv2.VideoCapture(f'{dir_recording}/world.mp4')

    df_fixations = pd.read_csv(fn_fixations)
    rows = list()
    for index,row in df_fixations.iterrows():
        print(f'Processing fixation {index}')
        row['ROI'] = processFixation(row, video)
        rows.append(row)

    df_result = pd.DataFrame(rows)
    df_result.to_excel('fixations.xlsx', index=False)
        


if __name__ == '__main__':
    main()