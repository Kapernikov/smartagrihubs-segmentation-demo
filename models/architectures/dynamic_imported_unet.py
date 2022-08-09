from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model


# Imported Model
def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)  # Normalize the values of the weights ( read more )
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPooling2D((2, 2))(x)
        return x, p
    else:
        return x


def build_unet(shape, num_classes, filters):
    print('Shape: ', shape)
    
    inputs = Input(shape)

    # Skip connections
    skips = []
    
    # Downsampling
    for (i, filter) in enumerate(filters[:-1]):
        # Input layer
        if i == 0:
            x, p = conv_block(inputs, filter, pool=True)
        
        else:
            x, p = conv_block(p, filter, pool=True)
        skips.append(x)
    
    # Bridge/Bottleneck
    p = conv_block(p, filters[-1], pool=False)
    
    """ Decoder """
    for (i,filter) in enumerate(reversed(filters[:-1])):
        # Hidden layers
        if i != len(filters[:-1]):
            p = UpSampling2D((2, 2), interpolation="bilinear")(p)
            skip = skips.pop()
            p = Concatenate()([p, skip])
            p = conv_block(p, filter, pool=False)
            p = Dropout(0.5)(p)
            
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(p)

    return Model(inputs, output)