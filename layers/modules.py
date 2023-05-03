from ray.rllib.utils import try_import_tf, try_import_tfp
from ray.rllib.utils.tf_ops import make_tf_callable


# python libraries
import os
import numpy as np


# tensorflow
tf = try_import_tf()
tfp = try_import_tfp()

if type(tf) == tuple:
    tf = tf[0]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#####################################################################
#####################################################################

# MAB / SAB / PMA layers for set transformers. This is a tensorflow adaptation to the Pycharm version from Lee et al 2019

class MAB(tf.keras.layers.Layer): # Multi-head self-attention layer
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=8, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        if dim_V % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {dim_V} should be divisible by number of heads = {num_heads}"
            )
        #self.projection_dim = embed_dim // num_heads
        self.fc_q = tf.keras.layers.Dense(dim_V)
        self.fc_k = tf.keras.layers.Dense(dim_V)
        self.fc_v = tf.keras.layers.Dense(dim_V)
        if ln:
            self.ln0 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
        self.fc_o = tf.keras.layers.Dense(dim_V)



    def call(self, Q, K):
        batch_size = tf.shape(K)[0]
        Q =  self.fc_q(Q)
        K,V = self.fc_k(K), self.fc_v(K)

        # Head splitting
        dim_split = self.dim_V // self.num_heads
        Q_ = tf.reshape(Q, (batch_size, -1, self.num_heads, dim_split))
        Q_ = tf.transpose(Q_, perm=[0, 2, 1, 3])   # (batch_size, num_heads, seq_len, projection_dim)
        K_ = tf.reshape(K, (batch_size, -1, self.num_heads, dim_split))
        K_ = tf.transpose(K_, perm=[0, 2, 1, 3])   # (batch_size, num_heads, seq_len, projection_dim)
        V_ = tf.reshape(V, (batch_size, -1, self.num_heads, dim_split))
        V_ = tf.transpose(V_, perm=[0, 2, 1, 3])   # (batch_size, num_heads, seq_len, projection_dim)

        # Attention
        score = tf.matmul(Q_, K_, transpose_b=True)
        dim_key = tf.cast(tf.shape(K)[-1], tf.float32)  # This should be K_ but set transformers implements with K
        scaled_score = score / tf.math.sqrt(dim_key)
        A = tf.nn.softmax(scaled_score, axis=-1)
        O = Q_ + tf.matmul(A, V_)  # (batch_size, num_heads, seq_len, projection_dim)

        O = tf.transpose(O, perm=[0, 2, 1, 3])   # (batch_size, seq_len, num_heads, projection_dim)
        O = tf.reshape(O, (batch_size, -1, self.dim_V))  # (batch_size, seq_len, embed_dim)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + tf.keras.activations.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB,self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def call(self, X):
        return self.mab(X,X)

class ISAB(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB,self).__init__()
        self.I = self.add_weight(shape=(1,num_inds, dim_out), initializer=tf.keras.initializers.glorot_uniform(), trainable=True)
        self.mab0 = self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def call(self,X):
        batch_size = tf.cast(tf.shape(X)[0], tf.int32)
        H = self.mab0(tf.tile(self.I,[batch_size, 1, 1]), X)
        return self.mab1(X, H)



class PMA(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = self.add_weight(shape=(1,num_seeds, dim), initializer=tf.keras.initializers.glorot_uniform(), trainable=True)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def call(self, X):
        batch_size = tf.cast(tf.shape(X)[0], tf.int32)
        return self.mab(tf.tile(self.S, [batch_size, 1, 1]), X)  # Repeat the seed vector for all the instances of the batch.
                                                # Dimensions (batch_size, num_seeds, dim)

