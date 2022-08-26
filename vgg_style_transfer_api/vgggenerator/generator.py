import tensorflow as tf
import time

from vgggenerator.model_builder import *
from vgggenerator.utils import *

### COUNT STYLE/CONTENT LOSS ###
def produce_targets(style_image, content_image):
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    return style_targets, content_targets

def style_content_loss(outputs, style_image, content_image,
                       style_weight=0.05, content_weight=0.1):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_targets, content_targets = produce_targets(style_image, content_image)
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                          for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                            for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def train_epoch(style_image, content_image,
                steps_per_epoch=100,
                style_weight=0.05,
                content_weight=0.1,
                total_variation_weight=0.2,
                image=None):


    # Check if we're passing a repeat image, or need to make a new one
    if tf.is_tensor(image):
      pass
    else:
      image = tf.Variable(content_image)

      # only generating the train_step function on first pass to reduce resource consumption

    @tf.function(experimental_relax_shapes=True)
    def train_step(image):
        with tf.GradientTape() as tape:
          outputs = extractor(image)
          loss = style_content_loss(outputs, style_image, content_image, style_weight=style_weight, content_weight=content_weight)
          loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))


    ### RUN MODEL ###
    start = time.time()

    step = 0
    for m in range(steps_per_epoch):
      step_start = time.time()
      step += 1
      train_step(image)
      step_end = time.time()
      print("Total time: {:.1f}".format(step_end-step_start))
      print("." * step, end='', flush=True)

    print("Train step: {}".format(step))
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return image
