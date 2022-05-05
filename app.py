from flask import Flask, render_template, request, flash, redirect, url_for
from keras.models import load_model
import numpy as np
import string
from PIL import Image
import random
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
import os

app = Flask(__name__)
app.secret_key = '4fd54u9svdjl43u9fdio54y;kl'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/images/uploads/')
model = load_model('model/improved_cnn_traffic_sign_recognition.h5')
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        image_path = 'images/dummy_image.jpg'
        return render_template("index.html", uploaded_image=image_path)

    if request.method == "POST":
        if 'sample_upload' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # Generating unique image name
        letters = string.ascii_lowercase
        name = ''.join(random.choice(letters) for i in range(10)) + '.png'
        uploaded_image = 'images/uploads/' + name

        # Reading, resizing, saving and preprocessing image for predicition
        image_upload = request.files['sample_upload']
        if image_upload and allowed_file(image_upload.filename):
            filename = secure_filename(image_upload.filename)
            image = Image.open(image_upload)
            image = image.resize((28, 28))
            image.save(os.path.join(UPLOADS_PATH, name))
            image_arr = np.array(image.convert('RGB'))
            image_arr.shape = (1, 28, 28, 3)

            # Predicting output
            result = model.predict(image_arr)
            class_num = np.argmax(result, axis=1)

            return render_template('index.html', uploaded_image=uploaded_image, class_label=classes[class_num[0]])
        else:
            return redirect(request.url)
        return redirect(request.url)


if __name__ == '__main__':
    app.debug = True
    app.run()
