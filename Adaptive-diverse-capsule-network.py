# -*- coding: utf-8 -*
"""
 This implement is an improved version of real-valued capsule network from our paper《Cv-CapsNet:complex-valued capsule network》,
 We introduce an attentional mechanism for fusing of three levels of features by weights so as to eliminate the manual setting
 of capsule dimensions in the coding stage.

 Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
 not just on CIFAR10.

 AUTHOR Jiangnan He  E-mail: 17375876392@csu.edu.cn, github:https://github.com/Johnnan002/Adaptive-diverse-capsule-network
 We refer to the implementation of the capsule network  Github: `https://github.com/XifengGuo/CapsNet-Keras`

Usage:
       python Adaptive-diverse-capsule-network.py
       ... ...
       
Result:
  Validation accuracy > 88.5% after 25 epochs.
  About 600 seconds per epoch on a single tesla k80 GPU card

"""
from keras.layers import Lambda
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length,bottleneck
from keras.layers.merge import concatenate
from capsulelayers import squash
K.set_image_data_format('channels_last')





def CapsNet(input_shape, n_class, routings):
    """
    Adaptive diverse Capsule Network on CIFAR10.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1


    x = layers.Input(shape=input_shape)
    l_1 = bottleneck(x, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_1 = concatenate([x, l_1], axis=concat_axis)

    l_2 = bottleneck(m_1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_2 = concatenate([m_1, l_2], axis=concat_axis)

    l_3 = bottleneck(m_2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_3 = concatenate([m_2, l_3], axis=concat_axis)

    l_4 =  bottleneck(m_3, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_4 = concatenate([m_3, l_4], axis=concat_axis)

    l_5 =  bottleneck(m_4, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_5 = concatenate([m_4, l_5], axis=concat_axis)

    l_6 =   bottleneck(m_5, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_6 = concatenate([m_5, l_6], axis=concat_axis)

    l_7 =   bottleneck(m_6, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_7 = concatenate([m_6, l_7], axis=concat_axis)

    l_8 =   bottleneck(m_7, 32, (3, 3), e=1, s=1, squeeze=False, nl='HS')
    m_8 = concatenate([m_7, l_8], axis=concat_axis)

    input1,primarycaps1,a1 = PrimaryCap(m_8,dim_capsule=12, n_channels=10, kernel_size=5, strides=(2,2), padding='valid')


    ################################################################################################


    l2_1=   bottleneck(input1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_1 = concatenate([input1, l2_1], axis=concat_axis)

    l2_2 =  bottleneck(m2_1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_2 = concatenate([m2_1, l2_2], axis=concat_axis)

    l2_3 =  bottleneck(m2_2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_3 = concatenate([m2_2, l2_3], axis=concat_axis)

    l2_4 =  bottleneck(m2_3, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_4 = concatenate([m2_3,l2_4], axis=concat_axis)

    l2_5 =  bottleneck(m2_4, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_5 = concatenate([m2_4, l2_5], axis=concat_axis)

    l2_6= bottleneck(m2_5, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_6 = concatenate([m2_5, l2_6], axis=concat_axis)

    l2_7 = bottleneck(m2_6, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_7 = concatenate([m2_6, l2_7], axis=concat_axis)

    l2_8 =  bottleneck(m2_7, 32, (3, 3), e=1, s=1, squeeze=False, nl='HS')
    m2_8 = concatenate([m2_7, l2_8], axis=concat_axis)

    input2, primarycaps2 ,a2= PrimaryCap(m2_8, dim_capsule=12, n_channels=10, kernel_size=5, strides=(2, 2), padding='valid')

################################################################################################
    l3_1 =  bottleneck(input2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_1 = concatenate([input2,  l3_1], axis=concat_axis)

    l3_2 =  bottleneck( m3_1 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_2 = concatenate([m3_1,l3_2], axis=concat_axis)

    l3_3 =  bottleneck( m3_2 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_3 = concatenate([m3_2,l3_3], axis=concat_axis)

    l3_4= bottleneck( m3_3 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_4 = concatenate([m3_3, l3_4], axis=concat_axis)

    l3_5 =  bottleneck( m3_4 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_5=concatenate([ m3_4, l3_5], axis=concat_axis)

    l3_6= bottleneck( m3_5 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_6=concatenate([m3_5, l3_6], axis=concat_axis)

    l3_7= bottleneck( m3_6 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_7=concatenate([m3_6, l3_7], axis=concat_axis)

    l3_8 =  bottleneck( m3_7 , 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_8 = concatenate([m3_7,l3_8], axis=concat_axis)

    input3, primarycaps3 ,a3= PrimaryCap(m3_8, dim_capsule=12, n_channels=10, kernel_size=3, strides=(1, 1), padding='valid')
    ################################################################################################


    primarycaps1=layers.Reshape(target_shape=(-1, 12), name='primarycaps11')(primarycaps1)
    primarycaps2=layers.Reshape(target_shape=(-1, 12), name='primarycaps21')(primarycaps2)
    primarycaps3 = layers.Reshape(target_shape=(-1,12), name='primarycaps31')(primarycaps3)

    digitcaps2= CapsuleLayer(num_capsule=n_class, dim_capsule=6, routings=routings,
                             name='digitcaps2')(primarycaps1)
    digitcaps3= CapsuleLayer(num_capsule=n_class, dim_capsule=6, routings=routings,
                             name='digitcaps3')(primarycaps2)

    digitcaps4= CapsuleLayer(num_capsule=n_class, dim_capsule=6, routings=routings,
                             name='digitcaps4')(primarycaps3)

    digitcaps2 = layers.Reshape(target_shape=(-1, 6), name='digitcaps21')(digitcaps2)
    a1 = K.tile(a1, [1,10, 6])
    weight_1 = Lambda(lambda x: x * a1)
    digitcaps2 = weight_1(digitcaps2)


    digitcaps3 = layers.Reshape(target_shape=(-1, 6), name='digitcaps31') (digitcaps3)
    a2= K.tile(a2, [1,10, 6])
    weight_2 = Lambda(lambda x: x * a2)
    digitcaps3=weight_2(digitcaps3)



    digitcaps4 = layers.Reshape(target_shape=(-1, 6), name='digitcaps41')(digitcaps4)
    a3 = K.tile(a3, [1,10, 6])
    weight_3 = Lambda(lambda x: x * a3)
    digitcaps4 = weight_3(digitcaps4)


    digitcaps = concatenate([ digitcaps2, digitcaps3], axis=-1)
    digitcaps = concatenate([ digitcaps, digitcaps4], axis=-1)

    digitcaps = layers.Lambda(squash)(digitcaps)
    out_caps = Length(name='capsnet')(digitcaps)
    y = layers.Input(shape=(n_class,))
    train_model = models.Model([x, y], out_caps)
    eval_model = models.Model(x, out_caps )
    return train_model, eval_model

####################################################capsnet################################


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=2)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                 loss=margin_loss,
                  loss_weights=[1.],
                 metrics={'capsnet': 'accuracy'})

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], y_batch)

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], y_test],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # callbacks=[log, tb, checkpoint])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    from utils1 import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)
    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred= model.predict(x_test, batch_size=128)
    print(y_pred)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)
    y_train = y_train.reshape(-1, 10)
    y_test = y_test.reshape(-1, 10)
    return  (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Adaptive Diverse Capsule Network on CIFAR10.")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.003, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.90, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = np.reshape(x_test, (-1, 32, 32, 3))

    model, eval_model= CapsNet(input_shape=x_train.shape[1:],
                                     n_class=10,routings=args.routings)
    model.summary()
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)
