from GAN_Train.GAN_model import make_generator_model, make_discriminator_model, generator_loss, discriminator_loss

import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras import optimizers


class Trainer:
    def __init__(self, batch_size=32, lr=1e-4, epochs=1000):
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
        for fname in [str(f.absolute()) for f in data_dir_path.glob('*.jpg')]:
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


    def save_model(self, model, epoch_num):
        model.save(f'gan_train_gutterdogs_E{epoch_num}', save_format='tf')
        print(f'saved gan_train_gutterdogs_E{epoch_num}')


    def pred_images_and_save(self, model, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(12, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(2, 3, i+1)
            plt.imshow(0.5*(predictions[i, :, :, :]+1))
            plt.axis('off')
        plt.savefig('Predicted_gutterdogs.jpg')
        print('saved Predicted_gutterdogs.jpg')


    def train(self):
        self.gen_loss_his = []
        self.disc_loss_his = []
        for epoch in range(self.epochs):

            for image_batch in self.train_dataset:
                self.train_step(image_batch)
                print('.')

            self.gen_loss_his.append(self.gen_loss)
            self.disc_loss_his.append(self.disc_loss)
            print(f'Epoch {epoch + 1} finished')

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1} finished')
                self.save_model(self.generator, epoch+1)

        # Generate after the final epoch
        print('Training finished, in total {self.epochs} epochs, now saving...')
        self.save_model(self.generator, self.epochs)
        self.pred_images_and_save(self.generator, self.seed)
        plt.clf()

        plt.plot(self.gen_loss_his, label='gen_loss')
        plt.plot(self.disc_loss_his, label='disc_loss')
        plt.legend()
        plt.savefig('Losses_gutterdogs.jpg')



if __name__ == "__main__":
    trainer = Trainer()
    print('Trainer ready')
    trainer.get_train_dataset('/home/jiayingliang1420/clean_gutterdogs_2946')
    trainer.launch_GAN()
    trainer.train()
