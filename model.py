import tensorflow as tf

DROPOUT = 0.5

def nn(input_features, out_dim, is_train=True):
    with tf.variable_scope('network', reuse=not is_train):
        h1 = tf.layers.dense(input_features, 
                             units=1024, 
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
        h1 = tf.layers.dropout(h1, rate=DROPOUT, training=is_train)
        
        h2 = tf.layers.dense(h1, 
                             units=1024, 
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
        h2 = tf.layers.dropout(h2, rate=DROPOUT, training=is_train)
        
        logits = tf.layers.dense(h2,
                              units=out_dim,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              activation=None)
        out = tf.nn.sigmoid(logits)
        return logits, out

    
def model_inputs(feature_size, label_size):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, feature_size], name="inputs") 
    labels = tf.placeholder(dtype=tf.float32, shape=[None, label_size], name="labels")
    lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
    return inputs, labels, lr

        
        