import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model

def depthwise_separable_conv(x, in_dim, out_dim, name):
    """Equivalent DepthwiseSeparableConv  PyTorch for channels-first."""
    x = SeparableConv2D(
        filters=out_dim,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        pointwise_initializer='glorot_uniform',
        data_format='channels_first',
        name=f'{name}_sepconv'
    )(x)
    x = BatchNormalization(axis=1, name=f'{name}_bn')(x)  # axis=1 for channels-first
    x = Activation('relu', name=f'{name}_relu')(x)
    return x

def single_2d_conv(x, in_dim, out_dim, name):
    """Equivalent Single2DConv  PyTorch."""
    return depthwise_separable_conv(x, in_dim, out_dim, name)

def VballNetFastV1(input_height, input_width, in_dim=9, out_dim=3):
    """
    Keras-\x80\x81 TrackNetV3NanoOptimized \x81 channels-first (N, C, H, W).

    Args:
        input_height (int): Input image height.
        input_width (int): Input image width.
        in_dim (int): Number of input channels ( \x83\x87\x8e 9 for \x82\x80\x85 RGB-\x80).
        out_dim (int): Number of output channels ( \x83\x87\x8e 3 for \x82\x80\x85 \x82\x8b\x85 \x80\x82).

    Returns:
        Model: Keras-\x8c VballNetFastV1.
    """
    inputs = Input(shape=(in_dim, input_height, input_width), name='input')  # (N, 9, H, W)

    # Encoder
    x1 = single_2d_conv(inputs, in_dim, 8, 'down_block_1')  # (N, 8, 288, 512)
    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        data_format='channels_first',
        name='pool1'
    )(x1)  # (N, 8, 144, 256)

    x2 = single_2d_conv(x, 8, 16, 'down_block_2')  # (N, 16, 144, 256)
    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        data_format='channels_first',
        name='pool2'
    )(x2)  # (N, 16, 72, 128)

    x3 = single_2d_conv(x, 16, 32, 'down_block_3')  # (N, 32, 72, 128)
    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        data_format='channels_first',
        name='pool3'
    )(x3)  # (N, 32, 36, 64)

    x = single_2d_conv(x, 32, 64, 'bottleneck')  # (N, 64, 36, 64)

    # Decoder
    x = UpSampling2D(
        size=(2, 2),
        interpolation='bilinear',
        data_format='channels_first',
        name='upsample1'
    )(x)  # (N, 64, 72, 128)
    x = Concatenate(axis=1, name='concat1')([x, x3])  # (N, 96, 72, 128)
    x = single_2d_conv(x, 96, 32, 'up_block_1')  # (N, 32, 72, 128)

    x = UpSampling2D(
        size=(2, 2),
        interpolation='bilinear',
        data_format='channels_first',
        name='upsample2'
    )(x)  # (N, 32, 144, 256)
    x = Concatenate(axis=1, name='concat2')([x, x2])  # (N, 48, 144, 256)
    x = single_2d_conv(x, 48, 16, 'up_block_2')  # (N, 16, 144, 256)

    x = UpSampling2D(
        size=(2, 2),
        interpolation='bilinear',
        data_format='channels_first',
        name='upsample3'
    )(x)  # (N, 16, 288, 512)
    x = Concatenate(axis=1, name='concat3')([x, x1])  # (N, 24, 288, 512)
    x = single_2d_conv(x, 24, 8, 'up_block_3')  # (N, 8, 288, 512)

    x = Conv2D(
        filters=out_dim,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='glorot_uniform',
        data_format='channels_first',
        name='predictor'
    )(x)  # (N, 3, 288, 512)
    outputs = Activation('sigmoid', name='sigmoid')(x)  # (N, 3, 288, 512)

    return Model(inputs=inputs, outputs=outputs, name='VballNetFastV1')