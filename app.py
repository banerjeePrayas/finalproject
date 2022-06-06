from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')

    image = cv2.imread('static/file.jpg', 0)

    if image is None:
        print('Wrong path:', image)
    else:
        image = cv2.resize(image, (48, 48))

        image = np.reshape(image, (1, 48, 48, 1))

    # loadedModel = pickle.load(open('.//final_model', 'rb'))

    # infile = open('modelfinal.h5', 'rb')
    # new_dict = pickle.load(infile)
    # infile.close()

    # print(new_dict)

    # console.log('Python')
    model = load_model('Case_1.h5')

    # loaded_model = pickle.load(open('./model', 'rb'))
    # model = pickle.load(open('./final_model', 'rb'))

    prediction = model.predict(image)

    label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('after.html', data=final_prediction)


if __name__ == '__main__':
    app.run(debug=True)
