### Function minimization with automatic differentiation and SGD ###
import tensorflow as tf

def dL_dx(x):
  '''Calculates derivative for the loss function using derivative formula'''
  return 2*(x-4)

# Initialize a random value for our initial x
x = tf.Variable([tf.random.normal([1])])
print("Initializing x={}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the loss, 
#   compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(500):
  with tf.GradientTape() as tape:
    '''TODO: define the loss as described above'''
    loss = (x - x_f)**2


  # loss minimization using gradient tape
  grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
  delta = learning_rate*grad
  new_x = x - delta # sgd update
  print(f'x:{x.numpy()[0]}')
  print(f'    d_loss/d_x  TAPE:[{grad}] | FORMULA:[{dL_dx(x.numpy()[0])}]')
  print(f'    delta: {delta} --> {new_x.numpy()[0]}')
  x.assign(new_x) # update the value of x
  history.append(x.numpy()[0])
