# import tensorflow_hub as hub
import tensorflow as tf
from vgggenerator.utils import gram_matrix, clip_0_1
from tensorflow.keras.models import save_model, load_model

# def download_model():
#   hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
#   return hub_model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
      print('### Started model building ###')
      super(StyleContentModel, self).__init__()
      self.vgg = build_vgg_layers(style_layers + content_layers)
      self.style_layers = style_layers
      self.content_layers = content_layers
      self.num_style_layers = len(style_layers)
      self.vgg.trainable = False

    def call(self, inputs):
      "Expects float input in [0,1]"
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

      style_outputs = [gram_matrix(style_output)
                      for style_output in style_outputs]

      content_dict = {content_name: value
                      for content_name, value
                      in zip(self.content_layers, content_outputs)}

      style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

      return {'content': content_dict, 'style': style_dict}

def build_vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    model.save('model')
    return model



  ### Define layers to use for content and style

content_layers = ['block1_conv1',
                'block1_conv2',
                'block1_pool',
                'block2_conv1',
                'block2_conv2',
                'block2_pool',
                'block3_conv1',
                'block3_conv2',
                'block3_conv3',
                'block3_conv4',
                'block3_pool',
                'block4_conv1']

style_layers = ['block1_conv1',
                'block1_conv2',
                'block1_pool',
                'block2_conv1',
                'block2_conv2',
                'block2_pool',
                'block3_conv1',
                              ]


num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

### BUILD MODEL WITH SELECTER LAYERS###
extractor = StyleContentModel(style_layers, content_layers)

### DEFINE MODEL OPTIMIZER ###
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)



### COUNT EDGE VARIATION LOSS ###
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
print('### model built ###')
### DEFINE TRAIN STEP ###

