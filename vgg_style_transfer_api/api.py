from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from vgggenerator.utils import load_img, tensor_to_image
from vgggenerator.generator import train_epoch
import io
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
import random
from PIL import Image

# create a fastapi instance
app = FastAPI()

from fastapi.responses import StreamingResponse
from google.cloud import storage

BUCKET_NAME = "wagoncn-850"
PATH_TO_GLC_MODEL = F'gs://{BUCKET_NAME}/images/'

def save_image_to_gcs(name):
  client = storage.Client().bucket(BUCKET_NAME)

  blob = client.blob(name)
  blob.upload_from_filename(name)
  print("=> image uploaded to bucket")
  os.remove(name)


async def generate_image(style_image, content_image,
                         image=None, epochs=10,
                         steps_per_epoch=3,
                         style_weight=0.05,
                         content_weight=0.1,
                         total_variation_weight=0.2):
    ### First time image is none, then it's the generated tensor
    for i in range(epochs):
        image = train_epoch(style_image, content_image,
          steps_per_epoch=steps_per_epoch,
          style_weight=style_weight,
          content_weight=content_weight,
          total_variation_weight=total_variation_weight,
          image=image)

        pil_image = tensor_to_image(image)
        # pil_image.save(f'latest_gen_{i}.png')
        # save_image_to_gcs(f'latest_gen_{i}.png')
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()
        yield img_byte_arr
    r = random.choice([150, 255])
    g = random.choice([150, 255])
    b = random.choice([150, 255])
    without_bg = Image.new("RGB", img_byte_arr.size, (r, g, b))
    without_bg.paste(img_byte_arr, mask=img_byte_arr.split()[3]) # 3 is the alpha channel




# create a predict endpoint that takes six arguments
@app.post("/generate")
async def generate( epochs=10,
                    steps_per_epoch=3,
                    style_weight=0.05,
                    content_weight=0.1,
                    total_variation_weight=0.25,
                    style_image: UploadFile=File(...),
                    content_image: UploadFile=File(...)):
  print('weights', style_weight, content_weight, total_variation_weight)

  # style image is real pet, content image is NFT
  # Taking bytes as input, turn it into tensor and reduce dimension to 512
  style_image = await style_image.read()
  content_image = await content_image.read()

  style_image = load_img(style_image)
  content_image = load_img(content_image)

  return StreamingResponse(generate_image(style_image, content_image,
                          epochs=int(epochs),
                          steps_per_epoch=int(steps_per_epoch),
                          style_weight=float(style_weight),
                          content_weight=float(content_weight),
                          total_variation_weight=float(total_variation_weight)),
                          media_type='image/jpeg')

#Increase api capacity
@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(20))

@app.get("/")
def home():
  return {'running': 'true'}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
