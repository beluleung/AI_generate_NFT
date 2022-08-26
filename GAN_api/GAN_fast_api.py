from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from skimage import exposure
import tensorflow as tf
import numpy as np
from PIL import Image

import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/guttercatgang")
def get_NFT_guttercatgang():
    trained_generator = tf.keras.models.load_model('GAN_api/GAN_guttercatgang')
    new_input = tf.random.normal([1, 100])
    pred = trained_generator(new_input, training=False)
    pred_img_rescaled = 0.5*(pred[0, :, :, :]+1)
    pred_img_rescaled = exposure.adjust_log(np.array(pred_img_rescaled), 2, inv=False)
    pred_img_rescaled = pred_img_rescaled*255
    img = Image.fromarray(pred_img_rescaled.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type='image/jpeg')

@app.get("/guttercatgangE400")
def get_NFT_guttercatgangE400():
    trained_generator = tf.keras.models.load_model('GAN_api/gan_train_guttercatgang_E400')
    new_input = tf.random.normal([1, 100])
    pred = trained_generator(new_input, training=False)
    pred_img_rescaled = 0.5*(pred[0, :, :, :]+1)
    pred_img_rescaled = exposure.adjust_log(np.array(pred_img_rescaled), 2, inv=False)
    pred_img_rescaled = pred_img_rescaled*255
    img = Image.fromarray(pred_img_rescaled.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type='image/jpeg')


@app.get("/gutterdogsE400")
def get_NFT_gutterdogsE400():
    trained_generator = tf.keras.models.load_model('GAN_api/gan_train_gutterdogs_E400')
    new_input = tf.random.normal([1, 100])
    pred = trained_generator(new_input, training=False)
    pred_img_rescaled = 0.5*(pred[0, :, :, :]+1)
    pred_img_rescaled = exposure.adjust_log(np.array(pred_img_rescaled), 2, inv=False)
    pred_img_rescaled = pred_img_rescaled*255
    img = Image.fromarray(pred_img_rescaled.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type='image/jpeg')

@app.get("/thedogepoundE450")
def get_NFT_thedogepoundE450():
    trained_generator = tf.keras.models.load_model('GAN_api/gan_train_thedogepound_E450')
    new_input = tf.random.normal([1, 100])
    pred = trained_generator(new_input, training=False)
    pred_img_rescaled = 0.5*(pred[0, :, :, :]+1)
    pred_img_rescaled = exposure.adjust_log(np.array(pred_img_rescaled), 2, inv=False)
    pred_img_rescaled = pred_img_rescaled*255
    img = Image.fromarray(pred_img_rescaled.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type='image/jpeg')
