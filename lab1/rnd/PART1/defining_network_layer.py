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
#tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = tf.constant([[1,2.]], shape=(1,2))
y = layer.call(x_input)

# test the output!
print(y.numpy())
mdl.lab1.test_custom_dense_layer_output(y)

'''
Arrays are not almost equal to 7 decimals
[FAIL] output is of incorrect value. expected [[0.16630773 0.49799666 0.4824747 ]] but got [[0.2697859  0.45750418 0.66536945]]
Mismatched elements: 3 / 3 (100%)
Max absolute difference: 0.18289474
Max relative difference: 0.38355663
 x: array([[0.1663077, 0.4979967, 0.4824747]], dtype=float32)
 y: array([[0.2697859, 0.4575042, 0.6653695]], dtype=float32)
  File "/Users/andc/Sandbox/2023/study/introtodeeplearning/idl-repo/lab1/rnd/defining_network_layer.py", line 39, in <module>
    mdl.lab1.test_custom_dense_layer_output(y)
AssertionError: 
Arrays are not almost equal to 7 decimals
[FAIL] output is of incorrect value. expected [[0.16630773 0.49799666 0.4824747 ]] but got [[0.2697859  0.45750418 0.66536945]]
Mismatched elements: 3 / 3 (100%)
Max absolute difference: 0.18289474
Max relative difference: 0.38355663
 x: array([[0.1663077, 0.4979967, 0.4824747]], dtype=float32)
 y: array([[0.2697859, 0.4575042, 0.6653695]], dtype=float32)
 '''