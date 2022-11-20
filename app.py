import os
import io
import cv2
import base64
import numpy as np
import sqlite3

from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from Prediction import prediction

# UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_db_connection():
    conn = sqlite3.connect('ocr.db')
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    conn = get_db_connection()
    cursor = conn.cursor()


    if 'file' not in request.files:
        return 'No File'

    file = request.files['file']

    if file.filename == '':
        return 'no file name'

    if file and allowed_file(file.filename):
        image_bytes = file.read()
        npimg = np.fromstring(image_bytes, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        # print(img)
        res_prediction = prediction(img)

        cursor.execute("INSERT INTO list_recognition (exp_date, product_code) VALUES (?, ?)",
                    (res_prediction[0], res_prediction[1])
                )

        conn.commit()
        conn.close()
        
        return jsonify(exp_date=res_prediction[0], product_code=res_prediction[1])
        # return 
    return ''
    # return 'hello world'
@app.route("/get_list_recognition", methods=["POST"])
def get_list_recognition():
    conn = get_db_connection()        
    results = conn.execute('SELECT * FROM list_recognition').fetchall()
    conn.close()

    list_recog = []
    for res in results:
        temp={
            'id': res[0],
            'exp_date': res[1],
            'product_code': res[2],
            'created_date': res[3]
        }
        list_recog.append(temp)

    return jsonify(result = list_recog)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

