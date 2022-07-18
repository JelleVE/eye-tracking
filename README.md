![example workflow](https://github.com/JelleVE/eye-tracking/actions/workflows/docker-publish.yml/badge.svg)

## Usage
* Clone this repository
* Install Docker
* Pull the Docker image using `docker pull ghcr.io/jelleve/eye-tracking:latest`
* Run a new Docker container using `docker run -d -p 5000:5000 ghcr.io/jelleve/eye-tracking:latest`
* Open the `launcher.html` file from this repository, select the relevant files and push the process button

## Notes
* Currently used: https://github.com/1adrianb/face-alignment (uses dlib)
* 3fabrec: https://github.com/browatbn2/3FabRec
* LFFD: https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices (no alignment?)
* PyLandmark: https://github.com/cnzeki/PyLandmark (no reference)
* TO INVESTIGATE: Retinaface

* TO CHECK 05/07: https://github.com/cunjian/pytorch_face_landmark
* TODO: check KPNet paper
* TODO: PIPNET -> Tested, fast
