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

    video_path = 'tmp.mp4'
    video.save(video_path)
    
    result = condition_assignment_pipnet.processFrames(video_path, pipnet)
    result = condition_assignment_pipnet.filterNoise(result, kernel_size=15)  # only interested in longer periods of closed eyes
    df_result = pd.DataFrame(result)
    df_result.to_excel('conditions_pipnet.xlsx', index=False)

    return send_from_directory('.', 'conditions_pipnet.xlsx', as_attachment=True)