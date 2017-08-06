import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

trainX = np.linspace(-1,1,101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")     # desired output

w = tf.Variable(0.0,name="weights")

y_model = tf.multiply(X,w)           # predicted output

cost = tf.pow(Y - y_model,2)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)

for i in range(100):
    for(x,y) in zip(trainX,trainY):
        _,cost_while_training=sess.run( [train_op,cost] , feed_dict = { X:x , Y:y } )
        #print(cost_while_training)
    print( sess.run(w) )




