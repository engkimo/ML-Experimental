# https://qiita.com/paperlefthand/items/82ab6df4a348f6070a55
#from flask_app import app
from flask import Flask, jsonify, request
import requests
import numpy as np
import cv2
import json
from gke_flask.predict import predict
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "Hello World!"

@app.route('/reply', methods=['GET', 'POST'])
def reply():
    if request.method == 'POST':
        data = request.data.decode('utf-8')
        url = json.loads(data)['key']
        resp = requests.get(url, stream=True).raw
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        pred_idx = predict(img)
        print(pred_idx)
        return str(pred_idx)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=8080,debug=True)
