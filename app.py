import cv2
import json
import pandas as pd

import condition_assignment_pipnet
import main_fixation_pipnet

from flask import Flask
from flask import request, send_from_directory

from PIPNet.pipnet import PIPNet

app = Flask(__name__)
pipnet = None



@app.before_first_request
def initialize():
    global pipnet
    pipnet = PIPNet()



@app.route("/process", methods=["GET", "POST"])
def process():
    video = request.files['video']
    fixations = request.files['fixations']

    video_path = 'tmp.mp4'
    video.save(video_path)
    print('Saved video')

    fixations_path = 'fixations.csv'
    fixations.save(fixations_path)
    print('Saved fixations')
    
    # conditions
    print('Processing conditions ...')
    condition_result = condition_assignment_pipnet.processFrames(video_path, pipnet)
    condition_result = condition_assignment_pipnet.filterNoise(condition_result, kernel_size=15)  # only interested in longer periods of closed eyes
    df_conditions = pd.DataFrame(condition_result)
    print('Processed conditions')
    
    # fixations
    print('Processing fixations ...')
    df_fixations = pd.read_csv(fixations_path)
    df_result = main_fixation_pipnet.processFixations(video_path, df_fixations, df_conditions, pipnet)
    df_result.to_excel('fixations_pipnet.xlsx')
    print('Processed fixations')

    return send_from_directory('.', 'fixations_pipnet.xlsx', as_attachment=True)