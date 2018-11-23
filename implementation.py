import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def import_dataset():
    (X_train,Y_train),(X_test,Y_test)=tf.keras.datasets.cifar10.load_data()
    X_train=X_train/255.0
    X_test=X_test/255.0
    Y_train=Y_train.reshape((Y_train.shape[0]))
    Y_test=Y_test.reshape((Y_test.shape[0]))
    Y_train=tf.one_hot(Y_train,depth=10)
    Y_test=tf.one_hot(Y_test,depth=10)
    with tf.Session() as sess:
        Y_train=sess.run(Y_train)
        Y_test=sess.run(Y_test)
    parameters={
        "X_train":X_train,
        "Y_train":Y_train,
        "X_test":X_test,
        "Y_test":Y_test
    }
    return parameters

def conv_net(X,is_training,dropout_rate,reuse):
  with tf.variable_scope('ConvNet', reuse=reuse):
    X=tf.reshape(X,[-1,32,32,3])
    A1=tf.layers.conv2d(X,64,3,strides=(1,1),padding='SAME',activation=tf.nn.relu)
    A2=tf.layers.conv2d(A1,128,5,strides=(1,1),padding='SAME',activation=tf.nn.relu)
    P1=tf.layers.max_pooling2d(A2,2,2)
    A2=tf.layers.conv2d(P1,256,3,strides=(1,1),padding='SAME',activation=tf.nn.relu)
    P2=tf.layers.max_pooling2d(A2,2,2)
    A3=tf.layers.conv2d(P2,512,3,strides=(1,1),padding='SAME',activation=tf.nn.relu)
    A4=tf.layers.conv2d(A3,1024,3,strides=(1,1),padding='SAME',activation=tf.nn.relu)
    P3=tf.layers.max_pooling2d(A4,2,2)
    P3=tf.contrib.layers.flatten(P3)
    dense1=tf.layers.dense(inputs=P3,units=2048,activation=tf.nn.relu)
    dropout1=tf.layers.dropout(inputs=dense1,rate=dropout_rate,training=is_training)
    dense2=tf.layers.dense(inputs=dropout1,units=1024,activation=tf.nn.relu)
    dropout2=tf.layers.dropout(inputs=dense2,rate=dropout_rate,training=is_training)
    logits=tf.layers.dense(inputs=dropout2,units=10)
    out=tf.nn.softmax(logits) if not is_training else logits
  return out

def main():
    epochs=1000
    minibatch_size=64
    parameters=import_dataset()
    X_train=parameters["X_train"]
    Y_train=parameters["Y_train"]
    X_test=parameters["X_test"]
    Y_test=parameters["Y_test"]
    m=X_train.shape[0]
    with tf.device("/gpu:0"):
        X=tf.placeholder(tf.float32,(None,32,32,3))
        Y=tf.placeholder(tf.float32,(None,10))
        conv_train=conv_net(X,is_training=True,dropout_rate=0.75,reuse=False)
        conv_test=conv_net(X,is_training=False,dropout_rate=0.75,reuse=True)
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=conv_train,labels=Y))
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
        grads=optimizer.compute_gradients(cost)
        train=optimizer.apply_gradients(grads)
        acc,acc_op=tf.metrics.accuracy(labels=tf.argmax(Y,1),predictions=tf.argmax(conv_train,1))
        correct_pred = tf.equal(tf.argmax(conv_test, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            costs=[]
            for epoch in range(epochs):
                minibatch_cost=0
                minibatch_accuracy=0
                for batch in range(0,m,minibatch_size):
                    minibatch_x=X_train[batch:batch+minibatch_size]
                    minibatch_y=Y_train[batch:batch+minibatch_size]
                    _,temp_cost=sess.run([train,cost],feed_dict={X:minibatch_x,Y:minibatch_y})
                    minibatch_accuracy+=(sess.run([acc_op],feed_dict={X:minibatch_x,Y:minibatch_y})[0])/int(m/minibatch_size)
                    minibatch_cost+=temp_cost/int(m/minibatch_size)
                costs.append(minibatch_cost)
                if epoch%10==0:
                  print("cost and accuracy after epoch %i: %f,%f"%(epoch,minibatch_cost,minibatch_accuracy))
            print("Testing Accuracy:", np.mean([sess.run(accuracy, feed_dict={X: X_test[i:i+minibatch_size],Y: Y_test[i:i+minibatch_size]}) for i in range(0,X_test.shape[0],minibatch_size)]))
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(0.001))
            plt.show()

main()