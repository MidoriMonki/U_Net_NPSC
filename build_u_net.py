from keras.models import Model
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization


def conv(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encode(input, num_filters):
    x = conv(input, num_filters)
    pool = MaxPool2D((2, 2))(x)

    return x, pool 

def decode(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv(x, num_filters)

    return x

def build_model(input_shape):
    inputs = Input(input_shape)
    
    #4 encoding layers, ends with 1024 features and image size of 14x14
    s1, p1 = encode(inputs, 64)
    s2, p2 = encode(p1, 128)
    s3, p3 = encode(p2, 256)
    s4, p4 = encode(p3, 512)

    b1 = conv(p4, 1024)

    #4 decoding layers, ends with 64 features and image size of 224x224
    d1 = decode(b1, s4, 512)
    d2 = decode(d1, s3, 256)
    d3 = decode(d2, s2, 128)
    d4 = decode(d3, s1, 64)

    #final convolutional layer to combine all the features
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return Model(inputs, outputs, name="U_Net")