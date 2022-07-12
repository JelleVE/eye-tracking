import requests
import json

url = 'http://localhost:5000/process'

with open('../eye-tracking-data/2021_01_24/001/world.mp4', 'rb') as vid:

    files = { 'video': vid }
    d = { 'body' : 'Foo Bar' }

    req = requests.post(url, files=files, json=d)

    print(req.status_code)
    print(req.text)