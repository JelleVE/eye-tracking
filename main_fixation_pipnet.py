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

from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
from matplotlib.patches import Ellipse

# import face_recognition
from PIPNet.pipnet import PIPNet


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



def processFrame(pipnet, fixation_id, frame_number, video, norm_pos_x, norm_pos_y, df_conditions, participant_id):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 360))

    # try:
    # image = face_recognition.load_image_file(f'frames/{current_datum["frame_number"]}.png')
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Draw gaze
    pos_x = int(round(image_width * norm_pos_x))
    pos_y = int(round(image_height * (1-norm_pos_y)))
    r = 5
    draw.ellipse((pos_x-r, pos_y-r, pos_x+r, pos_y+r), width=3)
    
    # Get cached data
    rows = df_conditions.loc[df_conditions['frame'] == frame_number]
    assert len(rows) == 1
    row = rows.iloc[0]
    if row['face_present'] == False:
        return False, False, False, False, False # early stopping

    # Draw landmarks
    #face_landmarks = pipnet.detectLandmarks(image, face_detections[0])
    # points_eyes = np.concatenate([face_landmarks['eye_left'], face_landmarks['eye_right'], face_landmarks['eyebrow_left'], face_landmarks['eyebrow_right']])
    # points_nose = face_landmarks['nose']
    # points_mouth = face_landmarks['lips']

    # Define points
    mean_left_eye = row['mean_left_eye']
    mean_right_eye = row['mean_right_eye']
    mean_nose = row['mean_nose']
    mean_mouth = row['mean_mouth']
    mean_points = np.vstack([mean_left_eye, mean_right_eye, mean_nose, mean_mouth])
    landmarks = row['face_landmarks']

    # Define upper ellipse
    # left = (int(landmarks['jaw'][0][0]), 
    #             int(landmarks['jaw'][0][1]))
    # right = (int(landmarks['jaw'][-1][0]),
    #             int(landmarks['jaw'][-1][1]))
    # # center = (int(landmarks['jaw'][0][0] + (landmarks['jaw'][32][0] - landmarks['jaw'][0][0])/2), 
    # #             int(landmarks['nose'][4][1]))
    # center = (int(landmarks['nose'][6][0]), 
    #             int(landmarks['nose'][6][1]))

    # Voronoi
    voronoi_polys = list()

    vor = Voronoi(mean_points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=1e4)

    min_x = 0
    max_x = image_width
    min_y = 0
    max_y = image_height

    mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
    bounded_vertices = np.max((vertices, mins), axis=0)
    maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
    bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    box = shapely.geometry.Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = shapely.geometry.Polygon(polygon)
        poly = poly.intersection(box)
        voronoi_polys.append(poly)
        

    # Create ROI: Assign Voronoi polygons their name and metadata
    regions = ['left_eye', 'right_eye', 'nose', 'mouth']
    mps = [mean_left_eye, mean_right_eye, mean_nose, mean_mouth]
    roi_mapping = dict()
    for index, mp in enumerate(mps):
        p = shapely.geometry.Point(mp)
        for poly in voronoi_polys:
            if p.intersects(poly):
                roi_mapping[regions[index]] = dict()
                roi_mapping[regions[index]]['voronoi_poly'] = poly
                roi_mapping[regions[index]]['mp'] = mp
                break

    # Create circles based on Voronoi diagram, with a given radius
    for region in roi_mapping.keys():
        relevant_poly = roi_mapping[region]['voronoi_poly']
        relevant_mp = roi_mapping[region]['mp']

        c = shapely.geometry.Point(relevant_mp).buffer(35)
        roi_mapping[region]['intersected_poly'] = c.intersection(relevant_poly)


    # Create lower ROI
    jaw_vertices = landmarks['jaw']
    #lower_vertices[0,:] = left # doesn't change anything
    #lower_vertices[-1,:] = right # doesn't change anything
    #lower_vertices = np.vstack([lower_vertices, center])
    jaw_poly = shapely.geometry.Polygon(jaw_vertices)


    # Create upper ROI
    eyes_poly = shapely.ops.unary_union([roi_mapping['left_eye']['intersected_poly'], roi_mapping['right_eye']['intersected_poly']])
    eyes_poly = eyes_poly.convex_hull
    eyes_poly = shapely.affinity.scale(eyes_poly, xfact=1.1, yfact=1.1, zfact=1.1, origin='center')

    # Perform additional early quitting
    # Check for "squishy face"
    #  https://github.com/JellinaP/faceMAP/issues/20
    dist = np.max(jaw_vertices[:,1]) - max(0, np.min(np.array(list(zip(*eyes_poly.exterior.coords.xy)))[:,1]))
    if dist < 100:
        return False, False, False, False, False

    # Create upper and lower ROI polygons
    combined_poly = shapely.ops.unary_union([jaw_poly, eyes_poly])
    combined_poly = combined_poly.convex_hull
    upper = eyes_poly
    upper_lower = combined_poly # combined instead of lower


    





    # upper_vertices = np.array(list(zip(*upper.exterior.coords.xy)))

    

    # Can probably be made more efficient
    # Idea: make a split between left and right, delete lower half
    #       of ellipse and find remaining lowest points (on both sides)
    #       these can then later be used to find the "split point"
    # centroid = upper.centroid
    # left_max_y = -1e10
    # right_max_y = -1e10
    # upper_list = list()
    # for row in upper_vertices:
    #     if row[0] <= centroid.x:
    #         if row[1] > left[1]-10:
    #             continue
    #         else:
    #             if row[1] > left_max_y:
    #                 left_max_y = row[1]
    #     elif row[0] > centroid.x:
    #         if row[1] > right[1]-10:
    #             continue
    #         else:
    #             if row[1] < right_max_y:
    #                 right_max_y = row[1]
    #     upper_list.append(row)
    
    # upper_vertices = np.array(upper_list)
    # mask = (upper_vertices[:, 0] < centroid.x) & (upper_vertices[:,1] == left_max_y)
    # assert np.sum(mask) == 1
    # index = np.where(mask==True)[0][0]

    # upper_a = upper_vertices[:index+1,:]
    # upper_c = upper_vertices[index+1:,:]
    # upper_b = np.array([left,center,right])
    # upper_vertices = np.vstack([upper_a, upper_b, upper_c])
    # upper = shapely.geometry.Polygon(upper_vertices)

    # Draw polys
    for region in roi_mapping.keys():
        poly = roi_mapping[region]['intersected_poly']
        coordinates = list(zip(*poly.exterior.coords.xy))
        for i in range(len(coordinates)-1):
            start = coordinates[i]
            stop = coordinates[i+1]
            draw.line([start, stop], width=3)

    # Draw upper
    poly = upper
    coordinates = list(zip(*poly.exterior.coords.xy))
    for i in range(len(coordinates)-1):
        start = coordinates[i]
        stop = coordinates[i+1]
        draw.line([start, stop], width=3, fill='red')
    
    # Draw lower
    poly = upper_lower
    coordinates = list(zip(*poly.exterior.coords.xy))
    for i in range(len(coordinates)-1):
        start = coordinates[i]
        stop = coordinates[i+1]
        draw.line([start, stop], width=3, fill='red')
    

    p_fixation = shapely.geometry.Point([pos_x, pos_y])

    bool_eyes = ('left_eye' in roi_mapping.keys() and p_fixation.intersects(roi_mapping['left_eye']['intersected_poly'])) or ('right_eye' in roi_mapping.keys() and p_fixation.intersects(roi_mapping['right_eye']['intersected_poly']))
    bool_nose = 'nose' in roi_mapping.keys() and p_fixation.intersects(roi_mapping['nose']['intersected_poly'])
    bool_mouth = 'mouth' in roi_mapping.keys() and p_fixation.intersects(roi_mapping['mouth']['intersected_poly'])
    bool_upper = p_fixation.intersects(upper)
    bool_lower = p_fixation.intersects(upper_lower) and not bool_upper

    fn_out = f'processed/{participant_id}/{fixation_id}/{frame_number}.png'
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    pil_image.save(fn_out)

    return bool_eyes, bool_nose, bool_mouth, bool_upper, bool_lower



def getFixationCondition(start_frame_index, end_frame_index, df_conditions):
    df_selection = df_conditions.loc[(df_conditions['frame'] >= start_frame_index) & (df_conditions['frame'] <= end_frame_index)]
    df_selection = df_selection.loc[df_selection['face_present'] == True]

    if len(df_selection) == 0: # early stopping
        return (False, None, -1, -1)
    
    new_start_frame_index = int(np.min(df_selection['frame']))
    new_end_frame_index = int(np.max(df_selection['frame']))

    # Majority rule
    conditions, counts = np.unique(df_selection['condition_processed'], return_counts=True)
    max_ind = np.argmax(counts)
    majority_condition = conditions[max_ind]
    
    return (True, majority_condition, new_start_frame_index, new_end_frame_index)



def processFixation(row, video_path, df_conditions, pipnet, global_frame_start, global_frame_end, participant_id):
    fixation_id = row['id']
    start_frame_index = row['start_frame_index']
    end_frame_index = row['end_frame_index']
    norm_pos_x = row['norm_pos_x']
    norm_pos_y = row['norm_pos_y']

    # early stopping cases
    if end_frame_index <= global_frame_start:
        return None
    if start_frame_index < global_frame_start and end_frame_index > global_frame_start:
        return None
    if start_frame_index >= global_frame_end:
        return None 
    if start_frame_index < global_frame_end and end_frame_index > global_frame_end:
        return None

    has_condition, majority_condition, new_start_frame_index, new_end_frame_index = getFixationCondition(start_frame_index, end_frame_index, df_conditions)
    if not has_condition: # early stopping
        return None    

    result_sequence = ['eyes', 'nose', 'mouth']
    result_sequence2 = ['upper', 'lower']
    result_counts = [0, 0, 0]
    result_counts2 = [0, 0]
    for current_frame_index in range(new_start_frame_index, new_end_frame_index+1):
        current_result = processFrame(pipnet, fixation_id, current_frame_index, video_path, norm_pos_x, norm_pos_y, df_conditions, participant_id)
        if current_result is not None:
            bool_eyes, bool_nose, bool_mouth, bool_upper, bool_lower = current_result
            result_counts[0] += bool_eyes
            result_counts[1] += bool_nose
            result_counts[2] += bool_mouth

            result_counts2[0] += bool_upper
            result_counts2[1] += bool_lower

    index_max = np.argmax(result_counts)
    if result_counts[index_max] == 0:
        row['AOI_Voronoi'] = 'none'
    else:
        row['AOI_Voronoi'] = result_sequence[index_max]

    # --

    index_max = np.argmax(result_counts2)
    if result_counts2[index_max] == 0:
        row['AOI_upper_lower'] = 'none'
    else:
        row['AOI_upper_lower'] = result_sequence2[index_max]

    row['start_frame_index'] = new_start_frame_index
    row['end_frame_index'] = new_end_frame_index
    row['condition'] = majority_condition

    return row



def processFixations(video_path, df_fixations, df_conditions, pipnet, global_frame_start, global_frame_end, participant_id):
    video = cv2.VideoCapture(video_path)

    rows = list()
    for index,row in df_fixations.iterrows():
        print(f'Processing fixation {index}')
        rows.append(processFixation(row, video, df_conditions, pipnet, global_frame_start, global_frame_end, participant_id))
    rows = [row for row in rows if row is not None]
    df_result = pd.DataFrame(rows)

    return df_result


def main():
    pipnet = PIPNet()
    dir_recording = '../eye-tracking-data/2021_01_24/001'
    fn_fixations = f'{dir_recording}/exports/001/fixations.csv'
    video = cv2.VideoCapture(f'{dir_recording}/world.mp4')
    df_conditions = pd.read_excel('conditions_pipnet.xlsx')

    df_fixations = pd.read_csv(fn_fixations, sep=';')
    rows = list()
    for index,row in df_fixations.iterrows():
        print(f'Processing fixation {index}')
        rows.append(processFixation(row, video, df_conditions, pipnet))

    rows = [row for row in rows if row is not None]
    df_result = pd.DataFrame(rows)
    df_result.to_excel('fixations_pipnet.xlsx', index=False)
        


if __name__ == '__main__':
    main()