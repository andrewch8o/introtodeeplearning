### Defining a network Layer ###
import tensorflow as tf
import mitdeeplearning as mdl


# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(OurDenseLayer, self).__init__()
    self.n_output_nodes = n_output_nodes

  def build(self, input_shape):
    d = int(input_shape[-1])
    # Define and initialize parameters: a weight matrix W and bias b
    # Note that parameter initialization is random!
    self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) # note the dimensionality
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

  def call(self, x):
    '''TODO: define the operation for z (hint: use tf.matmul)'''
    z = tf.matmul(x, self.W) + self.b

    '''TODO: define the operation for out (hint: use tf.sigmoid)'''
    y = tf.sigmoid(z)
    return y

# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.random.set_seed(1)
#tf.keras.utils.set_random_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))

print('👇W')
print(layer.W)
print('\n👇b')
print(layer.b)

x_input = tf.constant([[1,2.]], shape=(1,2))
print('\n👇x')
print(x_input)

print('\n👇result')
print(layer.call(x_input))
