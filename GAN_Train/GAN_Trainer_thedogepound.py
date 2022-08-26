from GAN_Train.GAN_model import make_generator_model, make_discriminator_model, generator_loss, discriminator_loss

import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras import optimizers


class Trainer:
    def __init__(self, batch_size=32, lr=1e-4, epochs=19):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.noise_dim = 100
        self.num_examples_to_generate = 6
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])


    def get_train_dataset(self, data_dir, img_height=320, img_width=320):
        print('Preparing the train dataset')
        data_dir_path = Path(data_dir)

        data = []
        for fname in [str(f.absolute()) for f in data_dir_path.glob('*.jpg')][:5000]:
            image = tf.io.read_file(fname)
            image = tf.io.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [img_height, img_width])
            image = (image*255 - 127.5) / 127.5
            data.append(image)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)

        print(f'In total {len(data)} images')
        print(f'{len(self.train_dataset)} batches per epoch')
        for batch in self.train_dataset:
            print(f'The shape of one batch: {batch.shape}')
            print(f'Normalization- Min: {tf.math.reduce_min(batch)}, Max: {tf.math.reduce_max(batch)}')
            break


    def launch_GAN(self):
        self.generator = make_generator_model()
        print('Generator built')

        self.discriminator = make_discriminator_model()
        print('Discriminator built')

        self.generator_optimizer = optimizers.Adam(self.lr)
        self.discriminator_optimizer = optimizers.Adam(self.lr)


    @tf.function
    def train_step(self, img_batch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.generator(noise, training=True)

          real_output = self.discriminator(img_batch, training=True)
          fake_output = self.discriminator(generated_images, training=True)

          self.gen_loss = generator_loss(fake_output)
          self.disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def save_generator(self, model, epoch_num):
        model.save(f's_gan_gen_thedogepound_E{epoch_num}', save_format='tf')
        print(f'saved gan_gen_thedogepound_E{epoch_num}')

    def save_discriminator(self, model, epoch_num):
        model.save(f's_gan_dis_thedogepound_E{epoch_num}', save_format='tf')
        print(f'saved gan_dis_thedogepound_E{epoch_num}')


    def pred_images_and_save(self, model, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(12, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(2, 3, i+1)
            plt.imshow(0.5*(predictions[i, :, :, :]+1))
            plt.axis('off')
        plt.savefig('s_Predicted_thedogepound.jpg')
        print('saved Predicted_thedogepound.jpg')


    def train(self):
        self.gen_loss_his = []
        self.disc_loss_his = []
        for epoch in range(self.epochs):

            for image_batch in self.train_dataset:
                self.train_step(image_batch)

            self.gen_loss_his.append(self.gen_loss)
            self.disc_loss_his.append(self.disc_loss)

            print(f'Epoch {epoch + 1} finished')

            self.save_generator(self.generator, epoch+1)


if __name__ == "__main__":
    trainer = Trainer()
    print('Trainer ready')
    trainer.get_train_dataset('/home/jiayingliang1420/clean_thedogepound_10k')
    trainer.launch_GAN()
    trainer.train()
