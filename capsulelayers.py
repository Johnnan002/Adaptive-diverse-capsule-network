# -*- coding: utf-8 -*
"""
 This implement is an improved version of real-valued capsule network from our paper《Cv-CapsNet:complex-valued capsule network》,
 We introduce an attentional mechanism for fusing of three levels of features by weights so as to eliminate the manual setting
 of capsule dimensions in the coding stage.

 Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
 not just on CIFAR10.

 AUTHOR Jiangnan He  E-mail: 17375876392@csu.edu.cn, github:https://github.com/Johnnan002/Adaptive-diverse-capsule-network
 We refer to the implementation of the capsule network  Github: `https://github.com/XifengGuo/CapsNet-Keras`

"""

import tensorflow as tf
from keras import initializers, layers
import numpy as np
weight_decay = 1E-4
from tflearn.layers.conv import global_avg_pool
from keras.layers import Conv2D,  Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Lambda
from keras.applications.mobilenet import DepthwiseConv2D
from keras import backend as K


K.set_image_data_format('channels_last')
concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

# Mobilenet V3 bottleneck 模块  
def bottleneck( inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.
        # Returns
            Output tensor.
        """
        def _relu6( x):
            """Relu 6
            """
            return K.relu(x, max_value=6.0)
        def _hard_swish( x):
            """Hard swish
            """
            return x * K.relu(x + 3.0, max_value=6.0) / 6.0
        
        def _return_activation(x, nl):
            """Convolution Block
            This function defines a activation choice.

            # Arguments
                x: Tensor, input tensor of conv layer.
                nl: String, nonlinearity activation type.

            # Returns
                Output tensor.
            """
            if nl == 'HS':
                x = Activation(_hard_swish)(x)
            if nl == 'RE':
                x = Activation(_relu6)(x)
            return x
           
        def _conv_block( inputs, filters, kernel, strides, nl):
            """Convolution Block
            This function defines a 2D convolution operation with BN and activation.

            # Arguments
                inputs: Tensor, input tensor of conv layer.
                filters: Integer, the dimensionality of the output space.
                kernel: An integer or tuple/list of 2 integers, specifying the
                    width and height of the 2D convolution window.
                strides: An integer or tuple/list of 2 integers,
                    specifying the strides of the convolution along the width and height.
                    Can be a single integer to specify the same value for
                    all spatial dimensions.
                nl: String, nonlinearity activation type.
            # Returns
                Output tensor.
            """
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
            x = BatchNormalization(axis=channel_axis)(x)
            return _return_activation(x, nl)

        def _squeeze( inputs):
            """Squeeze and Excitation.
            This function defines a squeeze structure.
            # Arguments
                inputs: Tensor, input tensor of conv layer.
            """
            input_channels = int(inputs.shape[-1])

            x = GlobalAveragePooling2D()(inputs)
            x = Dense(input_channels, activation='relu')(x)
            x = Dense(input_channels, activation='hard_sigmoid')(x)
            return x
           
           
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)
        tchannel = input_shape[channel_axis] * e
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        if squeeze:
            x = Lambda(lambda x: x * _squeeze(x))(x)
        x = _return_activation(x, nl)
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        return x



       
       
def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')
def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)
def Relu(x):
    return tf.nn.relu(x)
def Sigmoid(x):
    return tf.nn.sigmoid(x)


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config



def squash(vectors, axis=-1):
    """
     #squash() 是对向量长度进行归一化，防止向量长度大于1 
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon()) # 分母中的 1，试试0.5
    return scale * vectors



class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')


        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]


        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].

        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]

            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    def Squeeze_excitation_layer( input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)#对spatial 取平均响应强度反映整个feature的激活程度
            excitation= layers.Dense( units=int(out_dim/ratio))(squeeze)
            excitation = layers.Activation('relu')(excitation)
            excitation = layers.Dense(units=out_dim)(excitation)
            excitation=layers.Reshape(target_shape=(-1, out_dim))(excitation)
            excitation=tf.reduce_mean( excitation,axis=-1,keep_dims=True)
            excitation = Sigmoid(excitation)
            excitation = layers.Reshape(target_shape=(-1, 1))(excitation)
            return excitation
           
    #这里利用strides=2 实现降采样 
    output = BatchNormalization(axis=-1)(inputs)
    output = layers.Activation('relu')(output)
    output1 = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
      )(output)


    data_size = int(inputs.get_shape()[1])
    strides = strides[0]
    data_size = int(np.floor((data_size - kernel_size)/strides +1))
 
    outputs = layers.Reshape(target_shape=(-1, dim_capsule))(output1)
    #outputs =  tf.reshape(output1,[-1,data_size,data_size,n_channels, dim_capsule ])
   
    # 胶囊网络向量长度其实反映采样对区域的一个激活强度 也是一种attention机制 
    #这里先对胶囊取长度再对channel-wise取mean  作为整个区域的激活强度
    length=K.sqrt(K.sum(K.square(outputs ), -1))
    length =layers.Reshape(target_shape=(data_size,data_size,n_channels))(length)
    #length=tf.reshape(length,[-1,data_size,data_size,n_channels ])
    #SE模块 学习权重系数a
    a=Squeeze_excitation_layer(input_x=length, out_dim=n_channels, ratio=5, layer_name="se")
    
    
    return output1,layers.Lambda(squash)(outputs),a

