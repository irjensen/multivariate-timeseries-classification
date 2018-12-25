import tensorflow as tf

DROPOUT = 0.5

def mlp(input_features, out_dim, is_train=True):
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

    
def mlp_inputs(feature_size, label_size):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, feature_size], name="inputs") 
    labels = tf.placeholder(dtype=tf.float32, shape=[None, label_size], name="labels")
    lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
    return inputs, labels, lr


def cnn1d(input_data, out_dim, is_train=True):
    def conv1d(inputs, filters):
        return tf.layers.conv1d(inputs=inputs,
                                filters=filters,
                                kernel_size=5,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.nn.relu)
    
    with tf.variable_scope('network', reuse=not is_train):
        input_data = [tf.expand_dims(data, axis=-1) for data in input_data]
        
        h1 = [conv1d(data, 64) for data in input_data]
        h1 = [tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=1) for layer in h1]
        
        h2 = [conv1d(layer, 128) for layer in h1]
        h2 = [tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=1) for layer in h2]
        
        h3 = [conv1d(layer, 256) for layer in h2]
        h3 = [tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=1) for layer in h3]
        
        h4 = [conv1d(layer, 512) for layer in h3]
        h4 = [tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=1) for layer in h4]
        
        logits = [tf.contrib.layers.flatten(inputs=layer) for layer in h4]
        logits = tf.concat(logits, axis = -1)
        logits = tf.layers.dense(logits,
                                 units=out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        out = tf.nn.sigmoid(logits)
        
        return logits, out
    
    

def cnn1d_inputs(data_sizes, label_size):
    inputs = [tf.placeholder(dtype=tf.float32, shape=[None, size], name="inputs") for size in data_sizes]
    labels = tf.placeholder(dtype=tf.float32, shape=[None, label_size], name="labels")
    lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
    return inputs, labels, lr
        

        
        