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
        print('no face')
        return False, False, False, False, False, False # early stopping


    # Define points
    mean_left_eye = row['mean_left_eye']
    mean_right_eye = row['mean_right_eye']
    mean_nose = row['mean_nose']
    mean_mouth = row['mean_mouth']
    mean_points = np.vstack([mean_left_eye, mean_right_eye, mean_nose, mean_mouth])
    landmarks = row['face_landmarks']

    # Draw points
    draw.ellipse((row['mean_left_eye'][0]-r, row['mean_left_eye'][1]-r, row['mean_left_eye'][0]+r, row['mean_left_eye'][1]+r), width=3, fill='red')
    draw.ellipse((row['mean_right_eye'][0]-r, row['mean_right_eye'][1]-r, row['mean_right_eye'][0]+r, row['mean_right_eye'][1]+r), width=3, fill='red')
    draw.ellipse((row['mean_nose'][0]-r, row['mean_nose'][1]-r, row['mean_nose'][0]+r, row['mean_nose'][1]+r), width=3, fill='red')
    draw.ellipse((row['mean_mouth'][0]-r, row['mean_mouth'][1]-r, row['mean_mouth'][0]+r, row['mean_mouth'][1]+r), width=3, fill='red')

    # Voronoi
    voronoi_polys = list()
    mean_points = mean_points
    vor = Voronoi(mean_points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=1e4)

    min_x = 0
    max_x = image_width
    min_y = 0
    max_y = image_height
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
        return False, False, False, False, False, False

    if np.linalg.norm(row['mean_right_eye'] - row['mean_left_eye']) < 30: # Probably faulty detections
        return False, False, False, False, False, False

    if np.cross(row['mean_right_eye']-row['mean_left_eye'],row['mean_nose']-row['mean_left_eye'])/np.linalg.norm(row['mean_right_eye']-row['mean_left_eye']) < 20: # distance from eye line to nose
        return False, False, False, False, False, False

    # if row['mean_nose'][1] - row['mean_left_eye'][1] < 30: # Probably faulty detections
    #     return False, False, False, False, False, False

    # if row['mean_nose'][1] - row['mean_right_eye'][1] < 30: # Probably faulty detections
    #     return False, False, False, False, False, False

    # Create upper and lower ROI polygons
    combined_poly = shapely.ops.unary_union([jaw_poly, eyes_poly])
    combined_poly = combined_poly.convex_hull
    upper = eyes_poly
    upper_lower = combined_poly # combined instead of lower


    # Create full face ROI
    n = 100
    height = 150
    x1, y1 = jaw_vertices[0]
    x2, y2 = jaw_vertices[-1]
    x_c, y_c = (x1 + x2)/2, (y1 + y2)/2
    a, b = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/2, height/2
    angle = np.arctan2(y2 - y1, x2 - x1)
    t = np.linspace(0, 2*np.pi, n)
    ellipse = np.array([a*np.cos(t), b*np.sin(t)])
    ellipse = ellipse + np.array([[x_c,y_c]]).T
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])  
    ellipse_rot = R @ ellipse
    face_poly = shapely.geometry.Polygon(ellipse.T)
    face_poly = shapely.affinity.scale(face_poly, xfact=1.02, yfact=1.02, zfact=1.02, origin='center')
    full_face = shapely.ops.unary_union([face_poly, jaw_poly])


    # Draw full face
    poly = full_face
    coordinates = list(zip(*poly.exterior.coords.xy))
    for i in range(len(coordinates)-1):
        start = coordinates[i]
        stop = coordinates[i+1]
        draw.line([start, stop], width=8, fill='green')

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
    bool_full_face = p_fixation.intersects(full_face) or bool_eyes or bool_nose or bool_mouth or bool_upper or bool_lower

    fn_out = f'processed/{participant_id}/{fixation_id}/{frame_number}.png'
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    pil_image.save(fn_out)

    return bool_eyes, bool_nose, bool_mouth, bool_upper, bool_lower, bool_full_face



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
    result_sequence3 = ['full_face']
    result_counts = [0, 0, 0]
    result_counts2 = [0, 0]
    result_counts3 = [0]
    for current_frame_index in range(new_start_frame_index, new_end_frame_index+1):
        current_result = processFrame(pipnet, fixation_id, current_frame_index, video_path, norm_pos_x, norm_pos_y, df_conditions, participant_id)
        if current_result is not None:
            bool_eyes, bool_nose, bool_mouth, bool_upper, bool_lower, bool_full_face = current_result
            result_counts[0] += bool_eyes
            result_counts[1] += bool_nose
            result_counts[2] += bool_mouth

            result_counts2[0] += bool_upper
            result_counts2[1] += bool_lower

            result_counts3[0] += bool_full_face

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

    # --

    index_max = np.argmax(result_counts3)
    if result_counts3[index_max] == 0:
        row['AOI_full_face'] = 'none'
    else:
        row['AOI_full_face'] = result_sequence3[index_max]

    row['start_frame_index'] = new_start_frame_index
    row['end_frame_index'] = new_end_frame_index
    row['condition'] = majority_condition

    return row



def processFixations(video_path, df_fixations, df_conditions, pipnet, global_frame_start, global_frame_end, participant_id):
    video = cv2.VideoCapture(video_path)

    rows = list()
    for index,row in df_fixations.iterrows():
        print(f'Processing fixation {row["id"]}')
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