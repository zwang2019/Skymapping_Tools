import tensorflow as tf
import math
import numpy as np
import os

import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.client import device_lib

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K

choose_gpu = []
run = 'CD4Subtypes-TestRun9'


def residual_block_unit(inputs,
                        num_filters=16,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        net (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  data_format='channels_first',
                  kernel_regularizer=l2(1e-4))

    net = inputs
    if conv_first:
        net = conv(net)
        print("Convolution name= ", net.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=",
              strides)
        if batch_normalization:
            net = BatchNormalization()(net)
            print("Batch normalisation")
        if activation is not None:
            net = Activation(activation)(net)
            print("Activation")
    else:
        if batch_normalization:
            net = BatchNormalization()(net)
            print("Batch normalisation")
        if activation is not None:
            net = Activation(activation)(net)
            print("Activation")
        net = conv(net)
        print("Convolution name= ", net.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=",
              strides)
    # conv.name = conv.name + '_' + str(kernel_size) + 'x' + str(kernel_size) + '_' + str(num_filters) + '_' + str(
    #    strides)  # can be exam later
    return net


def resnet_v1(input_shape, depth, num_classes=4):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    x = 1

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    net = residual_block_unit(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            short_cut = residual_block_unit(inputs=net,
                                            num_filters=num_filters,
                                            strides=strides)
            short_cut = residual_block_unit(inputs=short_cut,
                                            num_filters=num_filters,
                                            activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                net = residual_block_unit(inputs=net,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
            net = tf.keras.layers.add([net, short_cut])
            net = Activation('relu')(net)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    net = AveragePooling2D(pool_size=8)(net)
    short_cut = Flatten()(net)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(short_cut)

    # the original version is y, but may cause some problem. Have to exam in the test.

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=4):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:777
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = residual_block_unit(inputs=inputs,
                            num_filters=num_filters_in,
                            conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = residual_block_unit(inputs=x,
                                    num_filters=num_filters_in,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=activation,
                                    batch_normalization=batch_normalization,
                                    conv_first=False)
            y = residual_block_unit(inputs=y,
                                    num_filters=num_filters_in,
                                    conv_first=False)
            y = residual_block_unit(inputs=y,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = residual_block_unit(inputs=x,
                                        num_filters=num_filters_out,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # why here use y instead of x ?

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 20, 30, 50 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 1e-4
    elif epoch > 55:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def config():
    print('''                                 Model parameter
         ----------------------------------------------------------------------------
                   |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
         Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
                   |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
         ----------------------------------------------------------------------------
         ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
         ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
         ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
         ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
         ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
         ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
         ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
         ---------------------------------------------------------------------------''')


def get_gpu(ggg):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for gpu in ggg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def let_tansel_know(str):
    print(str)


def train(X_train, X_test, X_validation, Y_train, Y_test, Y_validation, version=2, depth=20, epochs=60):

    strategy = tf.distribute.MirroredStrategy()
    if choose_gpu is not []:

        # devices:
        get_gpu(choose_gpu)
        print(_get_available_devices())
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    gpus = len(choose_gpu)

    tensorboard = TensorBoard(log_dir="log-full/{}".format(time.time()))

    # Training parameters
    if gpus == 0:
        batch_size = 32
    else:
        batch_size = 32 * gpus  # multiply by number of GPUs
    data_augmentation = False

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Model name, depth and version
    model_type = 'UResNet%dv%d' % (depth, version)
    print('model_type=', model_type)
    # Input image dimensions.

    input_shape = X_train.shape[1:]
    print('input_shape=', input_shape)

    # Normalize data.
    x_train = X_train.astype('float32') / 173
    x_test = X_test.astype('float32') / 173
    x_validation = X_validation.astype('float32') / 173

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        print(x_train_mean)
        x_train -= x_train_mean
        x_test -= x_train_mean
        x_validation -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_validation.shape[0], 'validation samples')
    print('y_train shape:', Y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = Y_train  # keras.utils.to_categorical(Y_train, num_classes)
    y_test = Y_test  # keras.utils.to_categorical(Y_test, num_classes)
    y_validation = Y_validation

    try:
        # if gpus > 1:
        #    with tf.device("/cpu:0"):
        #        if version == 2:
        #            model = resnet_v2(input_shape=input_shape, depth=depth)
        #        else:
        #            model = resnet_v1(input_shape=input_shape, depth=depth)
        # else:
        #    if version == 2:
        #        model = resnet_v2(input_shape=input_shape, depth=depth)
        #    else:
        #        model = resnet_v1(input_shape=input_shape, depth=depth)
        # model.summary()
        #    plot_model(model, to_file='model_plot.svg', show_shapes=True, show_layer_names=True)
        # if gpus > 1:
        #    model = multi_gpu_model(model, gpus=gpus, cpu_merge=False)

        if gpus >= 1:
            with strategy.scope():
                if version == 2:
                    model = resnet_v2(input_shape=input_shape, depth=depth)
                else:
                    model = resnet_v1(input_shape=input_shape, depth=depth)

                model.summary()

                model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=lr_schedule(0)),
                              metrics=['accuracy'])

        else:
            if version == 2:
                model = resnet_v2(input_shape=input_shape, depth=depth)
            else:
                model = resnet_v1(input_shape=input_shape, depth=depth)

            model.summary()

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=lr_schedule(0)),
                          metrics=['accuracy'])

        print(model_type)

        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = run + '_newdata_sc_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [tensorboard, checkpoint, lr_reducer, lr_scheduler]

        # Run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                #              steps_per_epoch=batch_size,
                                validation_data=(x_test, y_test),
                                #              validation_steps=1,
                                shuffle=True,
                                callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                # samplewise_s1td_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=0,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=False,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, meadata = adata[:, adata.var['highly_variable']]adata = adata[:, adata.var['highly_variable']]an, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                epochs=epochs, verbose=1, workers=4,
                                steps_per_epoch=batch_size,
                                callbacks=callbacks)
    except Exception as e:
        print('Error:', str(e))
        let_tansel_know("Training error=" + str(e)[0:1300])
    let_tansel_know("Training finished")

# wen ti shi na na ge gpu de wen ti zhao dao letr
# xian zai yao jie jue nage baocun de wen ti.
