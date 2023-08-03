import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, nodes, activation, l2=0):
        super(DenseBlock, self).__init__()

        layers = self.dense_block(nodes, activation, l2)
        self.dense, self.batchnorm, self.activation = layers

    def dense_block(self, nodes, activation, l2):
        name_prefix = 'FeedForward/DenseBlock/'

        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense'),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation),
                  )
        return layers


    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batchnorm(x)
        output = self.activation(x)
        return output


    
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, nodes, activation, l2=0):
        super(ResidualBlock, self).__init__()

        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation(activation, )
        self.dense1 = tf.keras.layers.Dense(nodes, 
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(l2))

        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation(activation)
        self.dense2 = tf.keras.layers.Dense(nodes, 
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(l2))
        self.add1 = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.batchnorm1(inputs)
        x = self.activation1(x)
        x = self.dense1(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.dense2(x)
        out = self.add1([x, inputs])
        return out



class ResidualNeuralNetwork(tf.keras.Model):

    def __init__(self, nodes, activations, l2=0):
        super(ResidualNeuralNetwork, self).__init__()
        
        # split nodes,activations into input, hidden and output parts for more readability
        input_node, hidden_nodes, output_node = nodes[0], nodes[1:-1], nodes[-1]
        input_activation, hidden_activations, output_activation = activations[0], activations[1:-1], activations[-1]

        # init network blocks
        # input blocks
        self.input_dense = tf.keras.layers.Dense(input_node)
        self.input_batchnorm = tf.keras.layers.BatchNormalization()
        self.input_activation = tf.keras.layers.Activation(input_activation)

        # hidden layers are residual blocks
        self.hidden_layers = [ResidualBlock(node, activation,l2)
                              for node, activation in zip(hidden_nodes, hidden_activations)]
        # output block
        self.output_dense = tf.keras.layers.Dense(output_node, activation=output_activation)

    def call(self, inputs):
        # input block
        x = self.input_dense(inputs)
        x = self.input_batchnorm(x)
        x = self.input_activation(x)
        # residual blocks
        for layer in self.hidden_layers:
            x = layer(x)

        output = self.output_dense(x)
        return output


if __name__ == '__main__':

    feed_forward_input = tf.ones(shape=(1, 128))

    feedforward_config = {'nodes': (128, 128, 128, 128), 
                          'activations': ('prelu', 'prelu', 'prelu', 'prelu'), 
                          'n_classes': 3, 
                          'l2':0.01}

    comb = ResidualNeuralNetwork(**feedforward_config)
    from IPython import embed
    embed()