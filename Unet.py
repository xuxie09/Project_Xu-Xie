
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import  BatchNormalization, Activation, average
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, BatchNormalization, ReLU, UpSampling2D, Add
from tensorflow.keras.models import Model
'''
#U-net
def Unet(img_rows,img_cols):
    num_channels = 1
    
    inputs = Input((img_rows, img_cols,num_channels))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
 
    up5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv8 = Conv2D(4, 1, activation = 'softmax')(conv7)
    model = Model(inputs = inputs, outputs = conv8)
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.99), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

'''
def conv_block(input, ch_out):
    x = Conv2D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def up_conv(input, ch_out):
    x = UpSampling2D(size=2)(input)
    x = Conv2D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def Attention_block(g, x, F_int):
    W_g = Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    W_g = BatchNormalization()(W_g)
    
    W_x = Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    W_x = BatchNormalization()(W_x)

    psi = ReLU()(Add()([W_g, W_x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization()(psi)
    psi = tf.keras.activations.sigmoid(psi)

    return x * psi

def Attention_U_Net(img_rows, img_cols, num_channels, output_channels):
    inputs = Input((img_rows, img_cols, num_channels))

    # Encoding path
    x1 = conv_block(inputs, 64)
    x2 = MaxPooling2D(pool_size=2, strides=2)(x1)
    x2 = conv_block(x2, 128)
    x3 = MaxPooling2D(pool_size=2, strides=2)(x2)
    x3 = conv_block(x3, 256)
    x4 = MaxPooling2D(pool_size=2, strides=2)(x3)
    x4 = conv_block(x4, 512)
    x5 = MaxPooling2D(pool_size=2, strides=2)(x4)
    x5 = conv_block(x5, 1024)

    # Decoding path
    d5 = up_conv(x5, 512)
    x4 = Attention_block(d5, x4, 256)
    d5 = concatenate([x4, d5])
    d5 = conv_block(d5, 512)

    d4 = up_conv(d5, 256)
    x3 = Attention_block(d4, x3, 128)
    d4 = concatenate([x3, d4])
    d4 = conv_block(d4, 256)

    d3 = up_conv(d4, 128)
    x2 = Attention_block(d3, x2, 64)
    d3 = concatenate([x2, d3])
    d3 = conv_block(d3, 128)

    d2 = up_conv(d3, 64)
    x1 = Attention_block(d2, x1, 32)
    d2 = concatenate([x1, d2])
    d2 = conv_block(d2, 64)

    outputs = Conv2D(output_channels, kernel_size=1, strides=1, padding='same', activation='softmax')(d2)

    model = Model(inputs, outputs)
    return model
