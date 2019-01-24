from data_generator import generate_data, plot_data, one_hot_encode
from keras.models import Model, load_model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, Concatenate, Input, Activation
import numpy as np

def generate_noise(n_samples, noise_dim):
    return np.random.normal(0, 1, size=(n_samples, noise_dim))

def generate_random_labels(labels_num, n):
    z = np.zeros((n, labels_num))
    x = np.random.choice(labels_num, n)
    for i in range(n):    
        z[i, x[i]] = 1    
    return z

def create_generator(input_noise_shape, input_condition_shape, output_img_shape):

    input_noise = Input(shape=input_noise_shape)
    input_condition = Input(shape=input_condition_shape)

    input_layer = Concatenate()([input_noise,input_condition])

    hid = Dense(128)(input_layer)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Dropout(0.3)(hid)

    hid = Dense(128)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Dropout(0.3)(hid)

    hid = Dense(output_img_shape)(hid)
    out = Activation('tanh')(hid)

    model = Model(inputs=[input_noise, input_condition], outputs=out)

    return model

def create_discriminator(input_condition_shape, input_img_shape):

    input_condition = Input(shape=input_condition_shape)
    input_image = Input(shape=(input_img_shape,))

    hid = Concatenate()([input_condition, input_image])

    hid = Dense(128)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Dropout(0.3)(hid)

    hid = Dense(128)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Dropout(0.3)(hid)

    hid = Dense(1)(hid)
    out = Activation('sigmoid')(hid)

    model = Model(inputs=[input_condition, input_image], outputs=out)

    return model

def create_GAN(noise_shape, img_shape, condition_shape):

    G = create_generator(noise_shape, condition_shape, img_shape)
    D = create_discriminator(condition_shape, img_shape)
    D.compile(optimizer='adam', loss='binary_crossentropy')
    #Note that D was compiled with D.trainable = True, which means calling D.train_on_batch(...)
    #will still adjust Ds weights, but calling GAN.train_on_batch(...) will not.
    D.trainable = False

    GAN_out = D([D.inputs[0],G([G.inputs[0],G.inputs[1]])])
    GAN = Model(inputs=[G.inputs[0],G.inputs[1],D.inputs[0]], outputs=GAN_out)
    GAN.compile(optimizer='adam', loss='binary_crossentropy')
    return GAN, G, D

def train_GAN(GAN, G, D, num_classes, num_epochs, num_batches, noise_shape, noise_factor):

    #Create loss tracking variables
    D_r_loss = []
    D_g_loss = []
    G_loss = []

    batch_size = int(len(X) / num_batches)

    # Train the GAN
    for epoch in range(num_epochs):
        for b in range(num_batches):

            # Train Discriminator on real images
            r_imgs = X[b*batch_size : (b+1)*batch_size]
            labels = Y[b*batch_size : (b+1)*batch_size]
            D_r_loss.append(D.train_on_batch([labels, r_imgs], np.ones((batch_size, 1))-noise_factor*np.random.random((batch_size,1))))

            # Train Discriminator on generated images
            noise = generate_noise(batch_size, noise_shape)
            random_labels = generate_random_labels(num_classes, batch_size)
            g_imgs = G.predict([noise, random_labels])
            D_g_loss.append(D.train_on_batch([random_labels, g_imgs],  np.zeros((batch_size, 1))+noise_factor*np.random.random((batch_size,1))))


              # Train Generator
            noise = generate_noise(batch_size, noise_shape)
            random_labels = generate_random_labels(num_classes, batch_size)
            G_loss.append(GAN.train_on_batch([noise, random_labels, random_labels], np.ones((batch_size, 1))))

            print(f'\rEpoch: {epoch+1}/{num_epochs}\tBatch: {b+1}/{num_batches}\tD_r_loss: {D_r_loss[-1]}, D_g_loss: {D_g_loss[-1]}, G_loss: {G_loss[-1]}', end='')
        print('')

seed = 10

num_classes = 5
num_features = 2
num_samples = 1000
num_epochs = 100
num_batches = 50
noise_factor = 0.1 #0.13

noise_shape = 100

X, Y_orig = generate_data(num_classes, num_features, num_samples, seed)
Y = one_hot_encode(Y_orig)

plot_data(X,Y_orig)

#Create Model
GAN, G, D = create_GAN((noise_shape,), num_features, (num_classes,))

train_GAN(GAN, G, D, num_classes, num_epochs, num_batches, noise_shape, noise_factor)

D.save('D')
G.save('G')
GAN.save('GAN')


