import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense,Conv2D, Reshape, Flatten,Conv2DTranspose,\
                         Lambda,MaxPooling2D, Dropout, concatenate, UpSampling2D,\
                         LeakyReLU, BatchNormalization, Add, Multiply

from keras.optimizers import Adam
from keras.applications import VGG19


def fs_rec_net(H=256,W=256,channels = 2,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))
    bn = BatchNormalization()(inputs)
    conv1 = Conv2D(64, kshape, activation='relu', padding='same')(bn)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(64, kshape, activation='relu', padding='same')(conv3)
    conv5 = Conv2D(64, kshape, activation='relu', padding='same')(conv4)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv5)
    conv7 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv8 = Conv2D(64, kshape, activation='relu', padding='same')(conv7)
    conv9 = Conv2D(64, kshape, activation='relu', padding='same')(conv8)
    conv10 = Conv2D(2,kshape, activation='linear', padding='same')(conv9)
    res1 = Add()([conv10,inputs])
    rec1 = Lambda(ifft_layer)(res1)
    rec2 = BatchNormalization()(rec1)
    conv11 = Conv2D(64, kshape, activation='relu', padding='same')(rec2)
    conv12 = Conv2D(64, kshape, activation='relu', padding='same')(conv11)
    conv13 = Conv2D(64, kshape, activation='relu', padding='same')(conv12)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(conv13)
    conv15 = Conv2D(64, kshape, activation='relu', padding='same')(conv14)
    conv16 = Conv2D(64, kshape, activation='relu', padding='same')(conv15)
    conv17 = Conv2D(64, kshape, activation='relu', padding='same')(conv16)
    conv18 = Conv2D(64, kshape, activation='relu', padding='same')(conv17)
    conv19 = Conv2D(64, kshape, activation='relu', padding='same')(conv18)
    conv20 = Conv2D(1, kshape, activation='linear', padding='same')(conv19)
    out = Add()([conv20,rec1])    
    model = Model(inputs=inputs, outputs=[res1,out])
    return model


def unet_rec(H=256,W=256,channels = 1,kshape = (5,5)):
    inputs = Input(shape=(H,W,channels))
    
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
    out = Add()([conv8,inputs])

    model = Model(inputs=inputs, outputs=out)
    return model


def fs_rec_unet(H=256,W=256,channels = 2,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))
    #bn = BatchNormalization()(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    res1 = Add()([conv8,inputs])
    rec1 = Lambda(ifft_layer)(res1)
    #rec2 = BatchNormalization()(rec1)
    
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(rec1)
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(up6)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(conv14)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv14)
    
    model = Model(inputs=inputs, outputs=[res1,out])
    return model

def fs_rec_unet_dc(H=256,W=256,channels = 2,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))
    #bn = BatchNormalization()(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    mask = Lambda(dc_layer)(inputs)
    dc = Multiply()([conv8,mask])
    res1 = Add()([dc,inputs])
    rec1 = Lambda(ifft_layer)(res1)
    
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(rec1)
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(up6)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(48, kshape, activation='relu', padding='same')(conv14)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv14)
    
    model = Model(inputs=inputs, outputs=[res1,out])
    return model

def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = K.tf.complex(real,imag)
    rec1 = K.tf.abs(K.tf.ifft2d(kspace_complex))
    rec1 = K.tf.expand_dims(rec1, -1)
    return rec1

def dc_layer(kspace):
    threshold = K.constant(0)
    mask = K.equal(kspace, threshold)
    mask = K.cast(mask, 'float32')
    return mask

def frequency_spatial_model(H=256,W=256,channels = 2,drop_out = 0.25,kshape = (3,3)):
    dropout = drop_out
    inputs = Input(shape=(H,W,channels))
    conv1 = Conv2D(32, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)
    
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(256, kshape, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, kshape, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(128, kshape, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, kshape, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(64, kshape, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, kshape, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv2D(32, kshape, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, kshape, activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(2, (1, 1), activation='linear')(conv9)
    rec1 = Lambda(ifft_layer)(conv10)
    
    conv11 = Conv2D(32, kshape, activation='relu', padding='same')(rec1)
    conv11 = Conv2D(32, kshape, activation='relu', padding='same')(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(64, kshape, activation='relu', padding='same')(pool11)
    conv12 = Conv2D(64, kshape, activation='relu', padding='same')(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(pool12)
    conv13 = Conv2D(128, kshape, activation='relu', padding='same')(conv13)
    pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    conv14 = Conv2D(256, kshape, activation='relu', padding='same')(pool13)
    conv14 = Conv2D(256, kshape, activation='relu', padding='same')(conv14)
    pool14 = MaxPooling2D(pool_size=(2, 2))(conv14)
    pool14 = Dropout(dropout)(pool14)
    
    conv15 = Conv2D(512, kshape, activation='relu', padding='same')(pool14)
    conv15 = Conv2D(512, kshape, activation='relu', padding='same')(conv15)
    
    up16 = concatenate([UpSampling2D(size=(2, 2))(conv15), conv14],axis=-1)
    up16 = Dropout(dropout)(up16)
    
    conv17 = Conv2D(256, kshape, activation='relu', padding='same')(up16)
    conv17 = Conv2D(256, kshape, activation='relu', padding='same')(conv17)
    
    up18 = concatenate([UpSampling2D(size=(2, 2))(conv17), conv13],axis=-1)
    up18 = Dropout(dropout)(up18)
    conv19 = Conv2D(128, kshape, activation='relu', padding='same')(up18)
    conv19 = Conv2D(128, kshape, activation='relu', padding='same')(conv19)
    
    up20 = concatenate([UpSampling2D(size=(2, 2))(conv19), conv12],axis=-1)
    up20 = Dropout(dropout)(up20)
    conv21 = Conv2D(64, kshape, activation='relu', padding='same')(up20)
    conv21 = Conv2D(64, kshape, activation='relu', padding='same')(conv21)
    
    up22 = concatenate([UpSampling2D(size=(2, 2))(conv21), conv11], axis=-1)
    up22 = Dropout(dropout)(up22)
    conv23 = Conv2D(32, kshape, activation='relu', padding='same')(up22)
    conv23 = Conv2D(32, kshape, activation='relu', padding='same')(conv23)
    
    conv24 = Conv2D(1, (1, 1), activation='linear')(conv23)
    model = Model(inputs=inputs, outputs=conv24)
    #opt = Adam()
    #model.compile(optimizer= opt,loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def unet_model(H=256,W=256,channels = 2,drop_out = 0.25,kshape = (3,3)):
    dropout = drop_out
    inputs = Input(shape=(H,W,channels))
    ifft = Lambda(ifft_layer)(inputs) 		
    conv1 = Conv2D(32, kshape, activation='relu', padding='same')(ifft)
    conv1 = Conv2D(32, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)
    
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(256, kshape, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, kshape, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(128, kshape, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, kshape, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(64, kshape, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, kshape, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv2D(32, kshape, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, kshape, activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)
    model = Model(inputs=inputs, outputs=con10)
    #opt = Adam()
    #model.compile(optimizer= opt,loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def build_vgg(img_shape):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]
    img = Input(shape=img_shape)
    # Extract image features
    img_features = vgg(img)
    return Model(img, img_features)

def build_discriminator(hr_shape,df):
    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    
    # Input img
    d0 = Input(shape=hr_shape)
    
    d1 = d_block(d0, df, bn=False)
    d2 = d_block(d1, df, strides=2)
    d3 = d_block(d2, df*2)
    d4 = d_block(d3, df*2, strides=2)
    d5 = d_block(d4, df*4)
    d6 = d_block(d5, df*4, strides=2)
    d7 = d_block(d6, df*8)
    d8 = d_block(d7, df*8, strides=2)
    
    d9 = Dense(df*16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(d0, validity)

