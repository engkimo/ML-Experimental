from flasks import app
from flask import Flask, request
import cv2
# curl localhost:5000/post -d '{"foo": "bar"}' -H 'Content-Type: application/json'

@app.route('/post', methods=['POST'])
def post_route():
    if request.method == 'POST':
        img = cv2.imread(request.get_data())
        return img



