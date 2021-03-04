import os
import tensorflow as tf
import numpy as np
import pandas as pd

def load_wind_model(sess, params, network_folder):

    saver = tf.compat.v1.train.import_meta_graph(os.path.join(network_folder, 'wind', params["model_name"]+".meta"))
    checkpoint = tf.compat.v1.train.latest_checkpoint(os.path.join(network_folder, 'wind'))
    saver.restore(sess, checkpoint)
    dcgan_model = ReplayedGAN(
        dim_y=params["n_events"],
        batch_size=params["batch_size"],
        image_shape=[params["n_gens"], params["n_timesteps_in_day"],1],
        tf_saver=saver,
        num_checkpoint=0
    )
    print("Replayed_W_DCGAN model loaded")

    return dcgan_model

def OneHot(X, n, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    for i in range(len(X)):
        m=X[i]
        Xoh[i,m]=1
    return Xoh

def generate_gaussian_inputs(params):
    n_preds = compute_n_preds(params)
    Y = []
    Z = []
    for pred in range(n_preds):
        Y.append(OneHot(np.random.randint(int(params["n_events"]), size=[params["batch_size"]]), n=int(params["n_events"])))
        Z.append(np.random.normal(params["mu"], params["sigma"], size=[int(params["batch_size"]), int(params["dim_inputs"])]).astype(np.float32))
    return Y,Z

def compute_n_preds(params):
    n_days_by_pred = params["batch_size"]
    n_days = (int(params["T"] / params["dt"]) // 24) + 1
    n_preds = (n_days // n_days_by_pred) + 1
    return n_preds

def post_process_sample(generated_batches, params, prods_charac):
    gens = prods_charac.columns
    wind = pd.DataFrame(columns = gens)

    for day, batch in enumerate(generated_batches):
        for i in range(params["batch_size"]):
            matrix = batch[i,:,:len(gens),0]
            df = pd.DataFrame(matrix, columns=gens)
            wind = pd.concat([wind, df], axis = 0)
    wind = wind.reset_index()
    wind = wind.iloc[:params['T'],:]
    return wind

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(o, t))

class ReplayedGAN():
    def __init__(
            self,
            batch_size=32,
            image_shape=[24,24,1],
            dim_z=100,
            dim_y=6, #The parameters for controlling the number of events
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            tf_saver = None,
            num_checkpoint = 0
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        # Read serialized network
        graph = tf.get_default_graph()
        self.gen_W1 = graph.get_tensor_by_name("gen_W1:"+str(num_checkpoint))
        self.gen_W2 = graph.get_tensor_by_name("gen_W2:"+str(num_checkpoint))
        self.gen_W3 = graph.get_tensor_by_name("gen_W3:"+str(num_checkpoint))
        self.gen_W4 = graph.get_tensor_by_name("gen_W4:"+str(num_checkpoint))

        self.discrim_W1 = graph.get_tensor_by_name("discrim_W1:"+str(num_checkpoint))
        self.discrim_W2 = graph.get_tensor_by_name("discrim_W2:"+str(num_checkpoint))
        self.discrim_W3 = graph.get_tensor_by_name("discrim_W3:"+str(num_checkpoint))
        self.discrim_W4 = graph.get_tensor_by_name("discrim_W4:"+str(num_checkpoint))


    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        h4 = self.generate(Z,Y)
        #image_gen comes from sigmoid output of generator
        image_gen = tf.nn.sigmoid(h4)

        raw_real2 = self.discriminate(image_real, Y)
        #p_real = tf.nn.sigmoid(raw_real)
        p_real=tf.reduce_mean(raw_real2)

        raw_gen2 = self.discriminate(image_gen, Y)
        #p_gen = tf.nn.sigmoid(raw_gen)
        p_gen = tf.reduce_mean(raw_gen2)

        discrim_cost = tf.reduce_sum(raw_real2) - tf.reduce_sum(raw_gen2)
        gen_cost = -tf.reduce_mean(raw_gen2)

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen


    def discriminate(self, image, Y):
        print("Initializing the discriminator")
        print("Y shape", Y.get_shape())
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        print("image shape", image.get_shape())
        print("yb shape", yb.get_shape())
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])],3)
        print("X shape", X.get_shape())
        h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        print("h1 shape", h1.get_shape())
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])],3)
        print("h1 shape", h1.get_shape())
        h2 = lrelu(batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        print("h2 shape", h2.get_shape())
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        discri=tf.matmul(h2, self.discrim_W3 )
        print("discri shape", discri.get_shape())
        h3 = lrelu(batchnormalize(discri))
        return h3


    def generate(self, Z, Y):
        print("Initializing the generator")
        print("Input Z shape", Z.get_shape())
        print("Input Y shape", Y.get_shape())
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat([Z,Y],1)
        print("Z shape", Z.get_shape())
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        print("h1 shape", h1.get_shape())
        h1 = tf.concat([h1, Y],1)
        print("h1 shape", h1.get_shape())
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        print("h2 shape", h2.get_shape())
        h2 = tf.reshape(h2, [self.batch_size,6,6,self.dim_W2])
        print("h2 shape", h2.get_shape())
        h2 = tf.concat([h2, yb*tf.ones([self.batch_size, 6,6, self.dim_y])],3)
        n=yb*tf.ones([self.batch_size, 6,6, self.dim_y])
        print("shape of yb new",n.get_shape() )
        print("h2 shape", h2.get_shape())

        output_shape_l3 = [self.batch_size,12,12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3))
        print("h3 shape", h3.get_shape())
        h3 = tf.concat([h3, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)
        print("h3 shape", h3.get_shape())

        output_shape_l4 = [self.batch_size,24,24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4


    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat([Z,Y], 1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,6,6,self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([batch_size, 6,6, self.dim_y])], 3)

        output_shape_l3 = [batch_size,12,12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat([h3, yb*tf.ones([batch_size, 12,12,self.dim_y])], 3)

        output_shape_l4 = [batch_size,24,24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z, Y, x