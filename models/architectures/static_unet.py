from keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose, Activation, MaxPooling2D, Concatenate
from keras.models import Model


# Implemented static model
def unet_static_model(shape, num_classes):
    inputs = Input(shape)
    print('input shape',inputs.shape)
    # Contracting path
    # Block 1 : Entry block
    c1 = Conv2D(16, 3, padding="same")(inputs)
    b1 = BatchNormalization()(c1)
    a1 = Activation("relu")(b1)

    c2 = Conv2D(16, 3, padding="same")(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation("relu")(b2)
    p1 = MaxPooling2D((2, 2), strides=2, padding="same")(a2)

    # Block 2
    c3 = Conv2D(32, 3, padding="same")(p1)
    b3 = BatchNormalization()(c3)
    a3 = Activation("relu")(b3)

    c4 = Conv2D(32, 3, padding="same")(a3)
    b4 = BatchNormalization()(c4)
    a4 = Activation("relu")(b4)
    p2 = MaxPooling2D((2, 2), strides=2, padding="same")(a4)

    # Block 3
    c5 = Conv2D(64, 3, padding="same")(p2)
    b5 = BatchNormalization()(c5)
    a5 = Activation("relu")(b5)

    c6 = Conv2D(64, 3, padding="same")(a5)
    b6 = BatchNormalization()(c6)
    a6 = Activation("relu")(b6)
    p3 = MaxPooling2D((2, 2), strides=2, padding="same")(a6)

    # Block 4
    c7 = Conv2D(128, 3, padding="same")(p3)
    b7 = BatchNormalization()(c7)
    a7 = Activation("relu")(b7)

    c8 = Conv2D(128, 3, padding="same")(a7)
    b8 = BatchNormalization()(c8)
    a8 = Activation("relu")(b8)
    p4 = MaxPooling2D((2, 2), strides=2, padding="same")(a8)

    # Bottleneck
    c9 = Conv2D(256, 3, padding="same")(p4)
    b9 = BatchNormalization()(c9)
    a9 = Activation("relu")(b9)
    c10 = Conv2D(256, 3, padding="same")(a9)
    b10 = BatchNormalization()(c10)
    a10 = Activation("relu")(b10)
    u1 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(a10)

    # Expansive path
    # Block 1
    conc1 = Concatenate()([u1, a8])
    c11 = Conv2D(128, 3, padding="same")(conc1)
    b11 = BatchNormalization()(c11)
    a11 = Activation("relu")(b11)

    c12 = Conv2D(128, 3, padding="same")(a11)
    b12 = BatchNormalization()(c12)
    a12 = Activation("relu")(b12)
    u2 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(a12)

    # Block 2
    conc2 = Concatenate()([u2, a6])
    c13 = Conv2D(64, 3, padding="same")(conc2)
    b13 = BatchNormalization()(c13)
    a13 = Activation("relu")(b13)

    c14 = Conv2D(64, 3, padding="same")(a13)
    b14 = BatchNormalization()(c14)
    a14 = Activation("relu")(b14)
    u3 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(a14)

    # Block 3
    conc3 = Concatenate()([u3, a4])
    c15 = Conv2D(32, 3, padding="same")(conc3)
    b15 = BatchNormalization()(c15)
    a15 = Activation("relu")(b15)

    c16 = Conv2D(32, 3, padding="same")(a15)
    b16 = BatchNormalization()(c16)
    a16 = Activation("relu")(b16)
    u4 = Conv2DTranspose(32, 2, strides=(2, 2), padding="same")(a16)

    # Block 4
    conc4 = Concatenate()([u4, a2])
    c17 = Conv2D(16, 3, padding="same")(conc4)
    b17 = BatchNormalization()(c17)
    a17 = Activation("relu")(b17)

    c18 = Conv2D(16, 3, padding="same")(a17)
    b18 = BatchNormalization()(c18)
    a18 = Activation("relu")(b18)

    # Output layer
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(a18)
    # Print outplut layer shape
    print("output.shape", output.shape)

    return Model(inputs, output)
