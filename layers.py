import tensorflow as tf
from tensorflow.keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='SAME', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        pool_size = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=pool_size,
            strides=strides,
            padding=self.padding,
            include_batch_in_index=True)
        return output, argmax

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // self.pool_size[0], input_shape[2] // self.pool_size[1], input_shape[3])

    def compute_mask(self, inputs, mask=None):
        return [None, None]

class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(None, default_name='unpool'):
            input_shape = tf.shape(updates, out_type=mask.dtype)
            output_shape = (input_shape[0],
                            input_shape[1] * self.pool_size[0],
                            input_shape[2] * self.pool_size[1],
                            input_shape[3])

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_size = tf.reduce_prod(output_shape)

            pool_size = self.pool_size[0] * self.pool_size[1]
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype), shape=[-1, 1, 1, 1])
            b = tf.ones_like(mask) * batch_range
            b = tf.reshape(b, [-1])
            m = tf.reshape(mask, [-1])
            f = tf.reshape(updates, [-1])

            updates_size = tf.size(f)
            indices = tf.stack([b, m], axis=1)
            indices = tf.reshape(indices, [updates_size, 2])

            ret = tf.scatter_nd(indices, f, [output_shape[0], flat_output_size // output_shape[0]])
            ret = tf.reshape(ret, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (mask_shape[0], mask_shape[1] * self.pool_size[0], mask_shape[2] * self.pool_size[1], mask_shape[3])
