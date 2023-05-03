# ray libraries
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
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

# custom layers
from layers.modules import SAB, ISAB, PMA


#####################################################################
#####################################################################
class zero_layer(tf.keras.layers.Layer):
    def __init__(self, encoder_dim=32, input_dim=32, name=""):
        super(zero_layer, self).__init__()
        w_init = tf.zeros_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, encoder_dim), dtype="float32"),
            trainable=False,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class custom_lstm_layer(tf.keras.layers.Layer):
    def __init__(self, encoder_dim):
        super(custom_lstm_layer, self).__init__()
        self.forget_gate = tf.keras.layers.Dense(encoder_dim, name="forget_gate", activation="sigmoid")
        self.input_gate = tf.keras.layers.Dense(encoder_dim, name="input_gate", activation="sigmoid")
        self.candidate_gate = tf.keras.layers.Dense(encoder_dim, name="candidate_values", activation="tanh")
        self.output_gate = tf.keras.layers.Dense(encoder_dim, name="output_gate", activation="sigmoid")

    def call(self, robot_info, h, c):
        concat_inputs = tf.keras.layers.concatenate([h,robot_info], axis=1)
        ft = self.forget_gate(concat_inputs)
        it = self.input_gate(concat_inputs)
        c_ = self.candidate_gate(concat_inputs)
        c = tf.math.add(tf.keras.layers.Multiply()([ft, c]), tf.keras.layers.Multiply()([it, c_]))
        ot = self.output_gate(concat_inputs)
        h = tf.keras.layers.Multiply()([ot, tf.keras.activations.tanh(c)])
        return h, c  #{"concat_inputs": concat_inputs, "ft": ft, "it": it, "c_": c_, "c": c, "ot": ot, "h": h}

def multi_robot_encoder(target_seq, encoder_dim, n_other_agents, child_dim):
    #offset = dim_p*2
    #dist_pos_vel_dim = dim_p*2+1
    input_dim = child_dim#n_obs_robots.shape[1]
    h_initializer = zero_layer(encoder_dim=encoder_dim, input_dim=input_dim, name="h_initial")
    c_initializer = zero_layer(encoder_dim=encoder_dim, input_dim=input_dim, name="c_initial")
    h_ini = h_initializer(target_seq[:,0]) #tf.keras.backend.zeros(shape=(encoder_dim,), name="h_initial")
    c_ini = c_initializer(target_seq[:,0])
    h = h_ini
    c = c_ini
    custom_lstm = custom_lstm_layer(encoder_dim)
    """
    forget_gate = tf.keras.layers.Dense(encoder_dim, name="forget_gate", activation="sigmoid")
    input_gate = tf.keras.layers.Dense(encoder_dim, name="input_gate", activation="sigmoid")
    candidate_gate = tf.keras.layers.Dense(encoder_dim, name="candidate_values", activation="tanh")
    output_gate = tf.keras.layers.Dense(encoder_dim, name="output_gate", activation="sigmoid")
    """
    debug_array = []
    for i in range(n_other_agents):
        robot_info = target_seq[:,i] #n_obs_robots[:,offset+i*dist_pos_vel_dim: offset+(i+1)*dist_pos_vel_dim]
        comparison = tf.equal(robot_info, tf.keras.backend.zeros_like(robot_info))
        h1, c1 = custom_lstm(robot_info, h, c)
        h = tf.where(comparison, x=h, y=h1, name="conditional_h")
        c = tf.where(comparison, x=c, y=c1, name="conditional_c")
        #debug_array.append(debug)

    encoding = h
    return encoding #, [h, c, h1, c1, comparison, robot_info[:,0], tf.keras.backend.zeros_like(robot_info[:,0])]



class LSTM_Encoder(TFModelV2):
    """Encoder-Transformer-Decoder network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        ## Heritage initialization
        super(LSTM_Encoder, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        ## Policy ID
        self.id = np.random.randint(1000)

        ## NETWORK HYPERPARAMETERS
        # Meta
        vf_share_layers = False #model_config.get("vf_share_layers") # Now it is True
        dim_p = model_config["custom_model_config"]["dim_p"]
        num_other_agents = model_config["custom_model_config"]["num_other_robots"]
        training = model_config["custom_model_config"]["training"]

        # Sequence building
        ntargets = model_config["custom_model_config"]["num_targets"]
        child_dim = obs_space.original_space.child_space['belief'].shape[0] + obs_space.original_space.child_space['measurement'].shape[0]+\
                    obs_space.original_space.child_space['location'].shape[0] + obs_space.original_space.child_space['velocity'].shape[0]+ \
                    obs_space.original_space.child_space['tracked'].shape[0]

        # Set Transformers
        num_inds = 32
        dim_hidden = 128  # 256
        num_heads = 4
        ln = True
        num_outputs = 6

        print('child_dim', child_dim)

        # Print configuration parameters
        print(obs_space, action_space, num_outputs, model_config)
        print(vf_share_layers, dim_p, num_other_agents, training)

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape), ), name="observations")

        # Configuration parameters
        batch_size = tf.shape(inputs)[0]

        ## LAYER DEFINITION
        #enc = tf.keras.Sequential([SAB(dim_hidden, dim_hidden, num_heads, ln=ln)])
        # dec = tf.keras.Sequential([PMA(dim_hidden, num_heads, 1,ln=ln),
        #                            tf.keras.layers.Dense(num_outputs, activation="tanh")])
        # dec_vf = tf.keras.Sequential([PMA(dim_hidden, num_heads, 1, ln=ln),
        #                               tf.keras.layers.Dense(1)])
        enc = tf.keras.Sequential([tf.keras.layers.Dense(dim_hidden, use_bias=False)])
        dec = tf.keras.Sequential([tf.keras.layers.Dense(num_outputs, activation="tanh")])
        dec_vf = tf.keras.Sequential([tf.keras.layers.Dense(1)])

        ## EXECUTION
        # Computing the input sequence
        #inputs_targets = inputs[:,1:ntargets*child_dim + 1] #tf.slice(inputs, [0,1], ) #inputs[:,1:]
        inputs_targets = inputs[:, 1:]
        target_seq = tf.reshape(inputs_targets, (batch_size, -1, child_dim))

        # Compute the architecture
        #encoded_seq = enc(target_seq)
        target_seq = enc(target_seq)
        encoded_seq = multi_robot_encoder(target_seq, dim_hidden, ntargets, dim_hidden) #child_dim
        action = dec(encoded_seq)
        action = tf.reshape(action, (batch_size, num_outputs))

        if not vf_share_layers:
            #enc_vf = tf.keras.Sequential([SAB(dim_hidden, dim_hidden, num_heads, ln=ln)])
            #encoded_seq_vf = enc_vf(target_seq)
            enc_vf = tf.keras.Sequential([tf.keras.layers.Dense(dim_hidden, use_bias=False)])
            target_seq = enc_vf(target_seq)
            encoded_seq_vf = multi_robot_encoder(target_seq, dim_hidden, ntargets, dim_hidden) #child_dim
        else:
            encoded_seq_vf = encoded_seq

        value_out = dec_vf(encoded_seq_vf)
        value_out = tf.reshape(value_out, (batch_size,1))

        ## MODEL DEFINITION
        self.base_model = tf.keras.Model(
            inputs, [action, value_out])
        self.output_inputs_model = tf.keras.Model(inputs, [inputs_targets])

        ## REGISTER VARIABLES FOR TRAINING
        self.register_variables(self.base_model.variables)

        ## PLOT SUMMARY OF THE MODEL
        tf.keras.utils.plot_model(self.base_model, "PPO_model.png")
        self.base_model.summary()


    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        #print("agent",self.id,":")
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def encoder_output(self, obs):
        #return self.encoder([obs])
        return self.transf_encoder([obs])

    def action_computation(self,obs):
        return self.base_model([obs])

    def output_inputs(self,obs):
        return self.output_inputs_model([obs])