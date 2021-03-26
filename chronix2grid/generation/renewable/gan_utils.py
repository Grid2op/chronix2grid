import os
import tensorflow as tf
import numpy as np
import pandas as pd

def load_model(sess, params, network_folder, carrier):

    saver = tf.compat.v1.train.import_meta_graph(os.path.join(network_folder, carrier, params["model_name_"+carrier]+".meta"))
    checkpoint = tf.compat.v1.train.latest_checkpoint(os.path.join(network_folder, carrier))
    saver.restore(sess, checkpoint)
    dcgan_model = ReplayedGAN(
        dim_y=params["n_events_"+carrier],
        batch_size=params["batch_size_"+carrier],
        image_shape=[params["n_timesteps_"+carrier],params["n_gens_"+carrier],1],
        dim_z=params["dim_inputs_"+carrier]
    )
    del saver
    print("Replayed_W_DCGAN model loaded for "+carrier)
    return dcgan_model, sess

def OneHot(X, n, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    for i in range(len(X)):
        m=X[i]
        Xoh[i,m]=1
    return Xoh

def generate_gaussian_inputs(params, carrier):
    n_preds = compute_n_preds(params, carrier)
    Y = []
    Z = []
    for pred in range(n_preds):
        Y.append(OneHot(np.random.randint(int(params["n_events_"+carrier]), size=[params["batch_size_"+carrier]]), n=int(params["n_events_"+carrier])))
        Z.append(np.random.normal(params["mu_"+carrier], params["sigma_"+carrier], size=[int(params["batch_size_"+carrier]), int(params["dim_inputs_"+carrier])]).astype(np.float32))
    return Y,Z

def compute_n_preds(params, carrier):
    n_days_by_pred = params["batch_size_"+carrier]
    n_days = (int(params["T"] / params["dt"]) // 24) + 1
    n_preds = (n_days // n_days_by_pred) + 1
    return n_preds

def post_process_sample(generated_batches, params, prods_charac, datetime_index, carrier = "wind"):
    gens = prods_charac[prods_charac['type']==carrier]["name"].unique()
    wind = pd.DataFrame(columns = gens)
    N = len(gens)
    N_learning = params['n_gens_'+carrier]

    if N > N_learning:
        raise ValueError("the neural network should be trained on at least the same number of generators as in the generation process")
    else:
        if N%2==0:
            selection_gens = [i for i in range(N//2)] + [i for i in range((N_learning-N//2), N_learning)]
        else:
            selection_gens = [i for i in range((N // 2)+1)] + [i for i in range((N_learning - N // 2), N_learning)]

    for day, batch in enumerate(generated_batches):
        for i in range(params["batch_size_"+carrier]):
            matrix = batch[i,:,selection_gens,0] # batch[i,:,:len(gens),0] # DIMENSIONS
            df = pd.DataFrame(np.transpose(matrix), columns=gens) # Enlever ou ajouter np.transpose en conséquence matric --> np.transpose(matrix)
            wind = pd.concat([wind, df], axis = 0)
    # Truncate last batch
    wind = wind.reset_index(drop=True)
    wind = wind.iloc[:len(datetime_index),:]

    # Time index
    wind['datetime'] = datetime_index

    # Power rescaling
    for gen in gens:
        Pmax = prods_charac.loc[prods_charac['name']==gen,'Pmax'].values[0]
        wind[gen] = wind[gen] * Pmax
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