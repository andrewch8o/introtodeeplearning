### Defining a model using subclassing and specifying custom behavior ###
import tensorflow as tf

class IdentityModel(tf.keras.Model):
  # As before, in __init__ we define the Model's layers
  # Since our desired behavior involves the forward pass, this part is unchanged
  def __init__(self, n_output_nodes):
    super(IdentityModel, self).__init__()
    self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

  '''TODO: Implement the behavior where the network outputs the input, unchanged, 
      under control of the isidentity argument.'''
  def call(self, inputs, isidentity=False):
    x = self.dense_layer(inputs)
    if isidentity:
        return x
    else:
        pass


n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))
'''TODO: pass the input into the model and call with and without the input identity option.'''
out_activate = model.call(inputs=x_input)
out_identity = model.call(inputs=x_input, isidentity=True)

#print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))