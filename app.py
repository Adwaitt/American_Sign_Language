from __future__ import division, print_function
import os
import tensorflow as tf
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import matplotlib.image as mpimg


model_path = r'C:\Users\HP\Desktop\work_comparision\vgg_model_final.h5'
model = tf.keras.models.load_model(model_path)

def image_predict(img):
    img = img/255
    img = img.reshape((1,) + img.shape)
    img = tf.image.resize(img, (224, 224), method = tf.image.ResizeMethod.BILINEAR)
    img = tf.convert_to_tensor(img)
    return np.argmax(model.predict(img))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        img = mpimg.imread(file_path)
        predict = str(image_predict(img))
        return predict
    return None

if __name__ == '__main__':
    app.run(debug=True)
