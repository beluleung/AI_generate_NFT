from tensorflow.keras import Input, Model, layers
import tensorflow as tf


def euc_criterion(in_, target):
    return tf.losses.mean_squared_error(target, in_)

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    # Note: this seems to be a mse criterion, not mae !
    return tf.reduce_mean((in_ - target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def init_discriminator(image_shape=(64, 64, 3), name='discriminator'):
    """Creates a discriminator taking one image and classifying it as a 'real' or 'fake' image
    
    Each layer consists of one Conv2D, one maxpool 2 to reduce feature map suize by a factor 2 and batchnorm
    """

    # image input for the correct shape
    img_in = Input(shape=image_shape, name='h-in') 

    # output of h1 is (batch_size x 32 x 32 x 16) for input images of shape (None, 64, 64, 3)
    h1 = layers.Conv2D(filters= 16, kernel_size=2, padding='same', activation='relu', name='h1_conv')(img_in)
    h1 = layers.MaxPool2D(pool_size=(2,2), name='h_maxpool1')(h1)
    h1 = layers.BatchNormalization(name='h_bn1')(h1)

    # output of h2 is (batch_size x 16 x 16 x 32) for input images of shape (None, 64, 64, 3)
    h2 = layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu',name='h2_conv')(h1)
    h2 = layers.MaxPool2D(pool_size=(2,2), name='h_maxpool2')(h2)
    h2 = layers.BatchNormalization(name='h_bn2')(h2)


    # output of h3 is (batch_size x 8 x 8 x 32) for input images of shape (None, 64, 64, 3)
    h3 = layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', name='h3_conv')(h2)
    h3 = layers.MaxPool2D(pool_size=(2,2), name='h_maxpool3')(h3)
    h3 = layers.BatchNormalization(name='h_bn3')(h3)

    # ouptput of h4 is (batch_size x 4 x 4 x 32) for input images of shape (None, 64, 64, 3)
    h4 = layers.Conv2D(filters=32,kernel_size=2, padding='same', activation='relu', name='h4_conv')(h3)
    h4 = layers.MaxPool2D(pool_size=(2,2), name='h_maxpool4')(h4)
    h4 = layers.BatchNormalization(name='h_bn4')(h4)

    # h5 is the classification output of shape (batch_size x 1)
    # Note that the sigmoid is in the loss function, this can be considered as the probability and uses linear activation
    h5 = layers.Flatten(name='h_flatten5')(h4)
    output = layers.Dense(units=1, activation='linear', name='h_output')(h5)

    # Create a Model object for this section of the X-GAN model
    discr = Model(img_in, output, name=name)

    return discr

def init_encoders(image_shape=(64, 64, 3)):
    """Creates two encoders: one for each domain (A and B)
    
    returns: enc_A and enc_B
    """
    
    # image input for the correct shape
    img_A = Input(shape=image_shape, name='e_in_A')
    img_B = Input(shape=image_shape, name='e_in_B')
  
    # Common layers for Encoder for domain A and Encoder for domain B
    # Last two Conv2D layers are common for both encoders
    common_conv3 = layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', name='e_common_conv3')
    common_conv4 = layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', name='e_common_conv4')
    # Fully Connected Layers to bring flatten CNN outputs to embeddings of shape (None, 1024)
    # Output shape is (None, 1024), providing the embeddings
    common_fc1 = layers.Dense(1024, activation='relu', name='e_common_fc1')
    common_fc2 = layers.Dense(1024, activation='relu', name='e_common_fc2')

    # Encoder for domain A
    # e1 is (batch_size x 32 x 32 x 32) for input images of shape (None, 64, 64, 3)
    eA1 = layers.Conv2D(32, kernel_size=2, padding='same', activation='relu', name='e_a_conv1')(img_A)
    eA1 = layers.MaxPool2D(pool_size=(2,2), name='eA_maxpool1')(eA1)
    eA1 = layers.BatchNormalization(name='e_bn1_A')(eA1)

    # e2 is (batch_size x 16 x 16 x 64) for input images of shape (None, 64, 64, 3)
    eA2 = layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',name='e_a_conv2')(eA1)
    eA2 = layers.MaxPool2D(pool_size=(2,2), name='e_a_maxpool2')(eA2)
    eA2 = layers.BatchNormalization(name='e_bn2_A')(eA2)

    # e3 is (batch_size x 8 x 8 x 128) for input images of shape (None, 64, 64, 3)
    eA3 = common_conv3(eA2)
    eA3 = layers.MaxPool2D(pool_size=(2,2), name='e_a_maxpool3')(eA3)
    eA3 = layers.BatchNormalization(name='eA_bn3')(eA3)

    # e4 is (batch_size x 4 x 4 x 256) for input images of shape (None, 64, 64, 3)
    eA4 = common_conv4(eA3)
    eA4 = layers.MaxPool2D(pool_size=(2,2), name='e_a_maxpool4')(eA4)
    eA4 = layers.BatchNormalization(name='eA_bn4')(eA4)
    
    # 2 Fully Connected Layers using the common layers
    # Output shape is (None, 1024), providing the embeddings
    eA5 = layers.Flatten(name='e_a_flat')(eA4)
    eA5 = common_fc1(eA5)
    output_A = common_fc2(eA5)

    enc_A = Model(img_A, output_A, name=f"enc-a")

    # Encoder for domain B
    # e1 is (batch_size x 32 x 32 x 32) for input images of shape (None, 64, 64, 3)
    eB1 = layers.Conv2D(32, kernel_size=2, padding='same', activation='relu', name='e_b_conv1')(img_B)
    eB1 = layers.MaxPool2D(pool_size=(2,2), name='e_b_maxpool1')(eB1)
    eB1 = layers.BatchNormalization(name='e_b_bn1')(eB1)

    # e2 is (batch_size x 16 x 16 x 64) for input images of shape (None, 64, 64, 3)
    eB2 = layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',name='e_b_conv2')(eB1)
    eB2 = layers.MaxPool2D(pool_size=(2,2), name='e_b_maxpool2')(eB2)
    eB2 = layers.BatchNormalization(name='eB_bn2')(eB2)

    
    # e3 is (batch_size x 8 x 8 x 128) for input images of shape (None, 64, 64, 3)
    eB3 = common_conv3(eB2)
    eB3 = layers.MaxPool2D(pool_size=(2,2), name='e_b_maxpool3')(eB3)
    eB3 = layers.BatchNormalization(name='e_b_bn3')(eB3)

    # e4 is (batch_size x 4 x 4 x 256) for input images of shape (None, 64, 64, 3)
    eB4 = common_conv4(eB3)
    eB4 = layers.MaxPool2D(pool_size=(2,2), name='e_b_maxpool4')(eB4)
    eB4 = layers.BatchNormalization(name='e_b_bn4')(eB4)


    # 2 Fully Connected Layers using the common layers
    # Output shape is (None, 1024), providing the embeddings
    eB5 = layers.Flatten(name='e_b_flat')(eB4)
    eB5 = common_fc1(eB5)
    output_B = common_fc2(eB5)

    enc_B = Model(img_B, output_B, name=f"enc-b")

    return enc_A, enc_B


def init_decoders():
    """Creates two decoders: one for each domain (A and B)
    
    returns: dec_A and dec_B
    """

    # Input shape is (1024, ), the shape of the embeddings
    in_A = Input(shape=(1024,))
    in_B = Input(shape=(1024,))

    common_deconv1 = layers.Conv2DTranspose(
        filters=512, kernel_size=(2, 2), strides=2, padding='same', 
        activation='relu', name='d_common_dconv1'
        )
    common_deconv2 = layers.Conv2DTranspose(
        filters=256, kernel_size=(2, 2), strides=2, padding='same', 
        activation='relu', name='d_common_dconv2'
        )

    dA1 = layers.Reshape(target_shape=(2, 2, 256))(in_A)
    dA1 = common_deconv1(dA1)
    dA1 = layers.BatchNormalization(name='d_a_btn1')(dA1)

    dA2 = common_deconv2(dA1)
    dA2 = layers.BatchNormalization(name='d_a_btn2')(dA2)

    dA3 = layers.Conv2DTranspose(
        filters=128, strides=2, kernel_size=(2,2), padding='same', 
        activation='relu', name='d_a_dconv3')(dA2)
    dA3 = layers.BatchNormalization(name='d_a_btn3')(dA3)

    dA4 = layers.Conv2DTranspose(
        filters=64, strides=2, kernel_size=(2,2), padding='same', 
        activation='relu', name='d_a_dconv4')(dA3)
    dA4 = layers.BatchNormalization(name='d_a_btn4')(dA4)

    dA5 = layers.Conv2DTranspose(
        filters=3, strides=2, kernel_size=(2,2), padding='same', 
        activation='tanh', name='d_a_dconv5')(dA4)
    dA5 = layers.BatchNormalization(name='d_a_btn5')(dA5)

    dec_A = Model(in_A, dA5, name='decoder-a')

    dB1 = layers.Reshape(target_shape=(2, 2, 256))(in_B)
    dB1 = common_deconv1(dB1)
    dB1 = layers.BatchNormalization(name='d_b_btn1')(dB1)

    dB2 = common_deconv2(dB1)
    dB2 = layers.BatchNormalization(name='d_b_btn2')(dB2)

    dB3 = layers.Conv2DTranspose(
        filters=128, strides=2, kernel_size=(2,2), padding='same', 
        activation='relu', name='d_b_dconv3')(dB2)
    dB3 = layers.BatchNormalization(name='d_b_btn3')(dB3)

    dB4 = layers.Conv2DTranspose(
        filters=64, strides=2, kernel_size=(2,2), padding='same', 
        activation='relu', name='d_b_dconv4')(dB3)
    dB4 = layers.BatchNormalization(name='d_b_btn4')(dB4)

    dB5 = layers.Conv2DTranspose(
        filters=3, strides=2, kernel_size=(2,2), padding='same', 
        activation='tanh', name='d_b_dconv5')(dB4)
    dB5 = layers.BatchNormalization(name='d_b_btn5')(dB5)

    dec_B = Model(in_B, dB5, name='decoder-b')

    return dec_A, dec_B

def init_cdann():
    """Creates the binary classifier Cdann, to classify "real" or "true" from embeddings

    Note: activation for the output is in the loss function, hence the linear activation in last layer

    Returns: the classifier model
    """
    
    embs_in = Input(shape=(1024, ), name='cdann-in')

    c1 = layers.Dense(1024, activation='relu', name='cdan-1')(embs_in)
    c2 = layers.Dense(256, activation='relu', name='cdan-2')(c1)
    c3 = layers.Dense(64, activation='relu', name='cdan-3')(c2)
    # output = layers.Dense(64, activation='softmax', name='cdan-out')(c3)
    output = layers.Dense(64, activation='linear', name='cdan-out')(c3)

    m = Model(embs_in, output, name='cdann')

    return m


def build_model(rec_weight=0.25, dann_weight=0.25, sem_weight=0.25, gan_weight=0.25):
    """Build a new xgan model"""

    if abs(1 - (rec_weight + dann_weight + sem_weight + gan_weight)) > 1e-3:
        raise ValueError(f"sum of all weights must be 1, not {rec_weight + dann_weight + sem_weight + gan_weight}")

    # Generator
    # Encoder output
    encA, encB = init_encoders()
    real_A = encA.input
    real_B = encB.input

    embedding_A = encA(real_A)
    embedding_B = encB(real_B)
    
    # Reconstruction output
    # A->encoderA->decoderA
    # B->encoderB->decoderB
    decA, decB = init_decoders()
    reconstruct_A = decA(embedding_A)
    reconstruct_B = decB(embedding_B)
        
    # Cdann output
    cdann = init_cdann()
    cdann_A = cdann(embedding_A)
    cdann_B = cdann(embedding_B)
    
    # Generator output
    # B->encoderB->decoderA
    # A->encoderA->decoderB
    fake_A = decA(embedding_B)
    fake_B = decB(embedding_A)
    
    # Fake image encoder output
    embedding_fake_A = encA(fake_A)
    embedding_fake_B = encB(fake_B)
    
    # Discriminator output
    discriminate_A = init_discriminator(name="discriminator_A")
    discriminate_B = init_discriminator(name="discriminator_B")
    discriminate_fake_A = discriminate_A(fake_A)
    discriminate_fake_B = discriminate_B(fake_B)


    # Loss
    # Reconstruction loss
    rec_loss_A = euc_criterion(real_A, reconstruct_A)
    rec_loss_B = euc_criterion(real_B, reconstruct_B)
    rec_loss = rec_loss_A + rec_loss_B

    # Domain-adversarial loss
    dann_loss = sce_criterion(cdann_A, tf.zeros_like(cdann_A)) + sce_criterion(cdann_B, tf.ones_like(cdann_B))

    # Semantic consistency loss
    sem_loss_A = abs_criterion(embedding_A, embedding_fake_B)
    sem_loss_B = abs_criterion(embedding_B, embedding_fake_A)
    sem_loss = sem_loss_A + sem_loss_B

    use_lsgan = True
    if use_lsgan:
        criterionGAN = mae_criterion
    else:
        criterionGAN = sce_criterion
        
    # Gan loss-generator part
    gen_gan_loss_A = criterionGAN(discriminate_fake_A, tf.ones_like(discriminate_fake_A))
    gen_gan_loss_B = criterionGAN(discriminate_fake_B, tf.ones_like(discriminate_fake_B))
    gen_gan_loss =  gen_gan_loss_A + gen_gan_loss_B

    # Total loss
    gen_loss = rec_weight * rec_loss + dann_weight * dann_loss + sem_weight * sem_loss + gan_weight * gen_gan_loss

    xgan = Model([real_A, real_B], [reconstruct_A, reconstruct_B], name='xgan')
    xgan.add_loss(gen_loss)
    xgan.add_metric(rec_loss_A, name='rec-a')
    xgan.add_metric(rec_loss_B, name='rec-b')
    xgan.add_metric(dann_loss, name='dann')
    xgan.add_metric(sem_loss, name='sem')
    xgan.add_metric(gen_gan_loss, name='gan')

    return xgan
