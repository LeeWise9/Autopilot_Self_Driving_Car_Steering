import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
# Fix error with Keras and TensorFlow
#import tensorflow as tf

#tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


# initialize parameter to record steering angle
# idx = 0
# steering_pred = []

@sio.on('telemetry')
def telemetry(sid, data):
    # global idx
    # global steering_pred
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # convert from BGR to RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # save frames locally
    # fname = 'test_img/fname' + str(idx) +'.jpg'
    # cv2.imwrite(fname, image_array)

    # resize image to match training data image size
    image_array = image_array[80:140, 0:320]
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5
    transformed_image_array = image_array[None, :, :, :]

    #
    # print(transformed_image_array.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # update steering angle record
    # steering_pred.append(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.8

    # idx = idx + 1
    # np.save('test_steering.npy', np.array(steering_pred))
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)