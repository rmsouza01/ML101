import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers import UpSampling2D, Dropout 
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    ''' Metric used for CNN training'''
    smooth = 1.0 #CNN dice coefficient smooth
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    ''' Loss function'''
    return -dice_coef(y_true, y_pred)

def get_unet_mod(patch_size = (None,None),learning_rate = 1e-5,\
                 learning_decay = 1e-6,gn_std = 0.025, drop_out = 0.25):
    ''' Get U-Net model with gaussian noise and dropout'''
    
    gaussian_noise_std = gn_std
    dropout = drop_out
    
    input_img = Input((patch_size[0], patch_size[1],3))
    input_with_noise = GaussianNoise(gaussian_noise_std)(input_img)    

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_with_noise)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_img, outputs=conv10)
    opt = Adam(lr= learning_rate, decay = learning_decay)
    model.compile(optimizer= opt,loss=dice_coef_loss, metrics=[dice_coef])

    return model

def pad_images(samples,nmaxpooling = 4):
    """
    This function pads an image so its dimensions are a multiple of 2**nmaxpooling 
    Input:
    samples -> 4D samples array (nsamples,W,Z,nchannels)
    nmaxpooling -> Number of maxpooling layers in the CNN
    Output:
    samples_padded -> 4D padded samples array	(nsamples,W+nw,Z+nz,nchannels)
    nw -> padding amount (channel 1)
    nz -> padding amount (channel 2)
    """
    nsamples,W,Z,nchannels = samples.shape
    nw = 2**nmaxpooling-W%(2**nmaxpooling)
    nz = 2**nmaxpooling-Z%(2**nmaxpooling)
    if nw == 0 and nz ==0:
        return samples,0,0
    samples_padded = np.zeros((nsamples,W+nw,Z+nz,nchannels))
    samples_padded[:,:-nw,:-nz,:] = samples
    return samples_padded,nw,nz                                
 
def rgb_images(img,period = None):
   """
   This function creates a 4D with 3 channel ("RGB") image from a time-resolved periodic image
   Input:
   img -> gray-scale 3D image (time,H,W)
   period -> number of time points in a period
   Output:
   img_rgb -> RGB image
   """                          
   t,H,W = img.shape
   if (period == None):
      period = t
        
   img_rgb = np.zeros((t,H,W,3))
   for ii in xrange(t/period):
      # Previous time point
      img_rgb[ii*period+1:(ii+1)*period,:,:,0] = img[ii*period:(ii+1)*period-1,:,:]
      # The previous time point for the first is the last 
      img_rgb[ii*period,:,:,0] = img[(ii+1)*period-1,:,:]
      # Current time point 
      img_rgb[ii*period:(ii+1)*period,:,:,1] = img[ii*period:(ii+1)*period,:,:]
      # Next time point 
      img_rgb[ii*period:(ii+1)*period-1,:,:,2] =  img[ii*period+1:(ii+1)*period,:,:]      # The next time point of the last is the first
      img_rgb[(ii+1)*period-1,:,:,2] =  img[ii*period,:,:]
   return img_rgb


