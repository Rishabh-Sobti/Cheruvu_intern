import pandas as pd
import numpy as np
train=pd.read_csv('training.csv')
test=pd.read_csv('sorted_test.csv')

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# Dividing into validation set
validation_features = train.iloc[0:115]
train = train.drop(range(0,116))

validation_labels = validation_features[['Ca','P','pH','SOC','Sand']].values
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
validation_features.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
predictedvals = ['Ca', 'P', 'pH', 'SOC', 'Sand']

# Encoding the 'depth' field and then applying Principal component analysis and taking only 10 features
# because more than 99% variance can be described by these 10 principal components
train = train.apply(LabelEncoder().fit_transform)
test = test.apply(LabelEncoder().fit_transform)
validation_features = validation_features.apply(LabelEncoder().fit_transform)
pca = PCA(n_components=10)
pca = pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)
validation_features = pca.transform(validation_features)

# Normalizing the principal components
train = normalize(train)
test = normalize(test)
validation_features = normalize(validation_features)

"""
Two SVR models with different C and epsilon
"""
clf1 = SVR(C=5000,epsilon=0.5)
clf1 = MultiOutputRegressor(clf1)
clf1 = clf1.fit(train, labels)
preds1 = clf1.predict(validation_features)

clf2 = SVR()
clf2 = MultiOutputRegressor(clf2)
clf2 = clf2.fit(train, labels)
preds2 = clf2.predict(validation_features)

# Calculating the rmse
import tensorflow as tf
tf.reset_default_graph()
logits = tf.placeholder(tf.float32, shape=(115,5))
y = tf.placeholder(tf.float32, shape=(115,5))
rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits,y)),0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rm = sess.run(rmse, feed_dict={y: validation_labels, logits: preds1})
    print('SVR_1')
    print(rm)
    rm = sess.run(rmse, feed_dict={y: validation_labels, logits: preds2})
    print('SVR_2')
    print(rm)
    

def neural_net_input(x_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32,shape=[None,x_shape],name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None,n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, shape=(None), name="keep_prob")
def layer(x_tensor, num_outputs, act='sigmoid'):
    """
    Applying a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    #shape after flattening
    flattened_shape = x_tensor.get_shape().as_list()[1]
    # weight and bias
    weights = tf.Variable(tf.truncated_normal([flattened_shape, num_outputs], stddev=0.04))
    bias = tf.Variable(tf.zeros([num_outputs]))
    
    # Fully connected layer
    if(act=='sigmoid'):
        fc = tf.nn.sigmoid(tf.add(tf.matmul(x_tensor, weights), bias))
    else:
        fc = tf.nn.relu(tf.add(tf.matmul(x_tensor, weights), bias))
    return fc

def output_layer(x_tensor, num_outputs):
    """
    Applying a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    #shape after flattening
    flattened_shape = x_tensor.get_shape().as_list()[1]
    # weight and bias
    weights = tf.Variable(tf.truncated_normal([flattened_shape, num_outputs], stddev=0.04))
    bias = tf.Variable(tf.zeros([num_outputs]))
    
    # Fully connected layer
    fc = tf.add(tf.matmul(x_tensor, weights), bias)
    return fc

def net1(x, keep_prob):
    out=layer(x, 5)
    out=tf.nn.dropout(out, keep_prob)
    out=layer(out, 5)
    out=tf.nn.dropout(out, keep_prob)
    out=output_layer(out, 5)
    return out
    
def net2(x, keep_prob):
    out=layer(x, 20)
    out=tf.nn.dropout(out, keep_prob)
    out=output_layer(out, 5)
    return out
    
def net3(x, keep_prob):
    out=layer(x, 5, 'relu')
    out=tf.nn.dropout(out, keep_prob)
    out=layer(out, 5, 'relu')
    out=tf.nn.dropout(out, keep_prob)
    out=output_layer(out, 5)
    return out
    
tf.reset_default_graph()

# Inputs
x = neural_net_input(10)
y = neural_net_label_input(5)
keep_prob = neural_net_keep_prob_input()

# Model
logits1 = net1(x, keep_prob)
logits2 = net2(x, keep_prob)
logits3 = net3(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits1 = tf.identity(logits1, name='logits1')
logits2 = tf.identity(logits2, name='logits2')
logits3 = tf.identity(logits3, name='logits3')

# Loss and Optimizer
cost1 = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.squared_difference(logits1,y),0)))
optimizer1 = tf.train.AdamOptimizer().minimize(cost1)
cost2 = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.squared_difference(logits2,y),0)))
optimizer2 = tf.train.AdamOptimizer().minimize(cost2)
cost3 = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.squared_difference(logits3,y),0)))
optimizer3 = tf.train.AdamOptimizer().minimize(cost3)

# Accuracy
rmse1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits1,y)),0), name='rmse1')
rmse2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits2,y)),0), name='rmse2')
rmse3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits3,y)),0), name='rmse3')

def train_neural_network(session, optimizer1, optimizer2, optimizer3, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer1, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability})
    session.run(optimizer2, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability})
    session.run(optimizer3, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability})
    
def print_stats(sess, rmse1, rmse2, rmse3):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    
    rmse_1 = sess.run(rmse1, feed_dict={x: validation_features,
                                       y: validation_labels,
                                       keep_prob: 1.})
    rmse_2 = sess.run(rmse2, feed_dict={x: validation_features,
                                       y: validation_labels,
                                       keep_prob: 1.})
    rmse_3 = sess.run(rmse3, feed_dict={x: validation_features,
                                       y: validation_labels,
                                       keep_prob: 1.})

    print('RMSE: Net 1: {} Net 2: {} Net 3: {}'.format(str(rmse_1), str(rmse_2), str(rmse_3)))
    


# Tune Parameters
epochs = 1000
keep_probability = 0.5

save_model_path = './cheruvu_models'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        train_neural_network(sess, optimizer1, optimizer2, optimizer3, keep_probability, train, labels)
        if(epoch%200==199):
            print('Epoch {:>2} '.format(epoch + 1), end='')
            print_stats(sess, rmse1, rmse2, rmse3)
    preds3=sess.run(logits1, feed_dict={x: validation_features, keep_prob: keep_probability})
    preds4=sess.run(logits2, feed_dict={x: validation_features, keep_prob: keep_probability})
    preds5=sess.run(logits3, feed_dict={x: validation_features, keep_prob: keep_probability})
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
final=np.empty(preds1.shape)
final[:,0] = 0.3*preds2[:,0] + 0.3*preds3[:,0] + 0.1*preds4[:,0] + 0.3*preds5[:,0]
final[:,1] = 0.1*preds2[:,1] + 0.3*preds3[:,1] + 0.3*preds4[:,1] + 0.3*preds5[:,1]
final[:,2] = 0.1*preds2[:,2] + 0.4*preds3[:,2] + 0.1*preds4[:,2] + 0.4*preds5[:,2]
final[:,3] = 0.8*preds1[:,3] + 0.1*preds4[:,3] + 0.1*preds5[:,3]
final[:,4] = 0.6*preds1[:,4] + 0.1*preds4[:,3] + 0.2*preds5[:,4]

tf.reset_default_graph()
logits = tf.placeholder(tf.float32, shape=(115,5))
y = tf.placeholder(tf.float32, shape=(115,5))
rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits,y)),0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rm = sess.run(rmse, feed_dict={y: validation_labels, logits: final})
    print('Final')
    print(rm)