from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model

# Operations 
class Conv_operation():
    def __init__(self, filters, batch_normalisation=True ,kernel_size=3):
        self.conv = Conv2D(filters, kernel_size, padding="same")
        if batch_normalisation:
            self.bn = BatchNormalization()
        self.activation = Activation("relu")
    
    def __call__(self, inputs):
        x = self.conv(inputs)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.activation(x)
        return x

# Blocks
class Downsampling_block():
    def __init__(self, filters, batch_normalisation=True):
        self.conv1 = Conv_operation(filters, batch_normalisation)
        self.conv2 = Conv_operation(filters, batch_normalisation)
        self.pool = MaxPooling2D((2, 2), strides=2, padding="same")
        
    def __call__(self, inputs):
        x = self.conv1(inputs)
        skip = self.conv2(x)
        pool = self.pool(x)
        return skip,pool

class BottleNeck_block():
    def __init__(self, filters,dropout_rate=0, batch_normalisation=True):    
        self.conv1 = Conv_operation(filters, batch_normalisation)
        self.conv2 = Conv_operation(filters, batch_normalisation)
        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate)
        self.conv_transpose = UpSampling2D((2, 2), interpolation="bilinear")
        #self.conv_transpose = Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")
    
    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv_transpose(x)
        return x

class Upsampling_block():
    def __init__(self, filters, batch_normalisation=True):
        self.conv1 = Conv_operation(filters, batch_normalisation)
        self.conv2 = Conv_operation(filters, batch_normalisation)
        self.drop = Dropout(0.5)
        # Dropout ?
        self.conv_transpose = UpSampling2D((2, 2), interpolation="bilinear")
        #self.conv_transpose = Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")

    def __call__(self, inputs, skip):
        x = Concatenate()([inputs, skip])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv_transpose(x)
        return x

class Output_block():
    def __init__(self, filters, batch_normalisation=True):
        self.conv1 = Conv_operation(filters, batch_normalisation)
        self.conv2 = Conv_operation(filters, batch_normalisation)

    def __call__(self, inputs, skip,num_classes):
        x = Concatenate()([inputs, skip])
        x = self.conv1(x)
        x = self.conv2(x)
        output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x)
        return output


# Unet Implementation
class UNet():
    def __init__(self,shape,num_classes,filters=[16,32,64,128,256],dropout_rate=0):
        self.shape = shape
        self.num_classes = num_classes
        self.filters = filters
        self.dropout_rate = dropout_rate
        #print("UNet with {} filters".format(filters))
    
    def get_model(self):
        inputs = Input(self.shape)
        
        # Skip connections
        skips = []

        # Downsampling blocks        
        for (i,filter) in enumerate(self.filters[:-1]):
            if i == 0:
                self.input_block = Downsampling_block(filter)
                skip,x = self.input_block(inputs)
            else:
                self.downsampling_block = Downsampling_block(filter)
                skip,x = self.downsampling_block(x)
            skips.append(skip)
        
        # Bottleneck block
        self.bottleneck_block = BottleNeck_block(self.filters[-1],self.dropout_rate)
        x = self.bottleneck_block(x)

        # Upsampling blocks
        for (i,filter) in enumerate(reversed(self.filters[:-1])):
            if i != len(self.filters[:-1])-1:
                self.upsampling_block = Upsampling_block(filter)
                skip = skips.pop()
                print("concatenating {} and {}".format(x.shape,skip.shape))
                x = self.upsampling_block(x, skip)
            else:
                self.output_block = Output_block(filter)
                skip = skips.pop()
                print("concatenating {} and {}".format(x.shape,skip.shape))
                output = self.output_block(x, skip, self.num_classes)
            
        return Model(inputs=inputs, outputs=output)
