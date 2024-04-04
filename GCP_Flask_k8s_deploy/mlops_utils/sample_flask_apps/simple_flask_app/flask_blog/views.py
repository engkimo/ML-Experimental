from flask_blog import app
from flask import request, redirect, url_for, render_template, flash, session
#import cv2
# curl localhost:5000/post -d '{"foo": "bar"}' -H 'Content-Type: application/json'
@app.route('/')
def show_entries():
    return render_template('entries/index.html')

