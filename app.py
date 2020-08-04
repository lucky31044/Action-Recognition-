import os
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import keras
import os.path
from tqdm import tqdm
import csv
import tensorflow
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
import cv2
from flask import Flask,render_template,request

app=Flask(__name__)

APP_ROOT =os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload",methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'static/')

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename=file.filename
        destination="/".join([target,filename])
        file.save(destination)
        abc=filename.split('.')
        global xyz
        xyz=abc[0]
    return render_template("complete.html",video_name=filename)

@app.route("/predict",methods=['POST','GET'])
def predict():
    temp=xyz
    os.chdir('static')
    uniquename = temp
    filetoread = uniquename+".mp4"
    # Read the video from specified path
    cam = cv2.VideoCapture(filetoread)
    os.chdir('..')
    try:

        # creating a folder named data
        if not os.path.exists(uniquename):
            os.makedirs(uniquename)

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # frame
    currentframe = 0

    while(True):

        # reading from frame
        ret,frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './'+uniquename+'/frame' + str(currentframe) + '.jpg'

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    seq_length= 40
    max_frames = 300
    image_shape=(224, 224, 3)
    base_path = uniquename
    classes = ['CricketBowling',
     'CricketShot',
     'FieldHockeyPenalty',
     'HandstandPushups',
     'HandstandWalking',
     'SoccerPenalty']
    data_clean = ['frame']
    def get_n_sample_from_video(sample, seq_length):
        path = os.path.join(base_path, sample[0], sample[1])
        filename = sample
        images = sorted(glob.glob(os.path.join(base_path, filename + '*jpg')))

        #Given a list and a size, return a rescaled/samples list. For example,
        #if we want a list of size 5 and we have a list of size 25, return a new
        #list of size five which is every 5th element of the origina list.
        # Get the number to skip between iterations.
        skip = len(images) // seq_length

        # Build our new output.
        output = [images[i] for i in range(0, len(images), skip)]

        # Cut off the last one if needed.
        return output[:seq_length]
    # Get model with pretrained weights.
    model = tensorflow.keras.models.load_model('inception.h5')

    def model_predict(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = model.predict(x)
        return features[0]
    os.makedirs('sequences', exist_ok=True)

    # Get the path to the sequence for this video.
    path = os.path.join('sequences', uniquename + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy

    # Get the frames for this video.
    frames = get_n_sample_from_video('frame', seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for frame in frames:
        features = model_predict(frame)
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)
    data= data_clean
    filename = uniquename
    X, y = [], []
    path = os.path.join('sequences' , filename + '-' + str(seq_length) +'-features.npy')
    sequence =np.load(path)
    X.append(sequence)
    X= np.array(X)
    nb_classes = len(classes)
    features_length = X.shape[-1]
    model = tensorflow.keras.models.load_model('video.h5')
    o=model.predict_classes(X)
    return render_template("output.html",display=o,video_name=filetoread)

if __name__ == "__main__ " :
    app.run(debug=True)
