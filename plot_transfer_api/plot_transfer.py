import plot_transfer_api.model as md
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response


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
    return {"status": "ok"}

@app.post('/plot-transfer')
async def plot_transfer(img: UploadFile=File(...), model_weights='plot_transfer_api/test_model_weights.h5'):
    # load model
    model = md.build_model()
    model.load_weights(model_weights)
    #Get encoder and decoder from model
    enc_B, dec_A = [i for i in model.layers if i.name in ['enc-b','decoder-a']]

    # change image to tensor
    #read_img = tf.io.read_file(img)
    read_img = await img.read()
    tensor = tf.io.decode_raw(read_img, tf.uint8)
    tensor2 = tf.io.decode_image(read_img, channels=3)
    img1 = tf.image.resize(tensor2, [64,64])
    img2 = tf.expand_dims(img1, axis=0)

    # transform test image
    new_nft = dec_A(enc_B(img2))
    tf.linalg.normalize(new_nft, ord=1)

    # image normalize
    from sklearn.preprocessing import MinMaxScaler

    flatten = tf.reshape(new_nft, (64*64,3))
    scaler = MinMaxScaler()
    flatten_scale = scaler.fit_transform(flatten).astype('float32')

    nnew_nft = tf.constant(flatten_scale.reshape([1, 64, 64, 3]))

    img_nparr=np.array(nnew_nft[0])*255
    img = Image.fromarray(img_nparr.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content = img_byte_arr, media_type="image/jpeg")
