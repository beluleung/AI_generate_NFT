import model as md
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def plot_transfer(user_input_img, model_weights='test_model_weights.h5'):
    # load model
    model = md.build_model()
    model.load_weights(model_weights)
    #Get encoder and decoder from model
    enc_B, dec_A = [i for i in model.layers if i.name in ['enc-b','decoder-a']]

    # change image to tensor
    img = tf.io.read_file(user_input_img)
    tensor = tf.io.decode_image(img, channels=3)
    img1 = tf.image.resize(tensor, [64,64])
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
    breakpoint()

    img_nparr=np.array(nnew_nft[0])*255
    img = Image.fromarray(img_nparr.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr
