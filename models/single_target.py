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


class FullyConnectedModel(TFModelV2):
    """Encoder-Transformer-Decoder network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        ## Heritage initialization
        super(FullyConnectedModel, self).__init__(
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
        # child_dim = obs_space.original_space["targets"]["all"].child_space['belief'].shape[0] + \
        #             obs_space.original_space["targets"]["all"].child_space['measurement'].shape[0] + \
        #             obs_space.original_space["targets"]["all"].child_space['location'].shape[0] + \
        #             obs_space.original_space["targets"]["all"].child_space['velocity'].shape[0] + \
        #             obs_space.original_space["targets"]["all"].child_space['tracked'].shape[0]

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

        ## INPUT RESHAPING
        # input_targets_all = tf.reshape(inputs, (batch_size, 3, -1))[:,-1,1:]
        # inputs_target = tf.reshape(input_targets_all, (batch_size,-1,child_dim))[:,0]

        ## LAYER DEFINITION
        layer1 = tf.keras.Sequential([tf.keras.layers.Dense(dim_hidden, activation="relu")])
        layer2 = tf.keras.Sequential([tf.keras.layers.Dense(num_outputs, activation="tanh")])
        layer2_vf = tf.keras.Sequential([tf.keras.layers.Dense(1)])

        ## EXECUTION
        # Computing the input sequence
        #inputs_targets = inputs[:,1:ntargets*child_dim + 1] #tf.slice(inputs, [0,1], ) #inputs[:,1:]
        inputs_target = inputs[:, 1:child_dim+1]
        target_seq = tf.reshape(inputs_target, (batch_size, 1, child_dim))

        # Compute the architecture
        encoded_seq = layer1(target_seq)
        action = layer2(encoded_seq)
        action = tf.reshape(action, (batch_size, num_outputs))

        if not vf_share_layers:
            layer1_vf = tf.keras.Sequential([tf.keras.layers.Dense(dim_hidden, activation="relu")])
            encoded_seq_vf = layer1_vf(target_seq)
        else:
            encoded_seq_vf = encoded_seq

        value_out = layer2_vf(encoded_seq_vf)
        value_out = tf.reshape(value_out, (batch_size,1))

        ## MODEL DEFINITION
        self.base_model = tf.keras.Model(
            inputs, [action, value_out])
        self.output_inputs_model = tf.keras.Model(inputs, [inputs_target])

        ## REGISTER VARIABLES FOR TRAINING
        self.register_variables(self.base_model.variables)

        ## PLOT SUMMARY OF THE MODEL
        # tf.keras.utils.plot_model(self.base_model, "PPO_model.png")
        self.base_model.summary()


    def forward(self, input_dict, state, seq_lens):
        #flattened_input = input_dict["obs"]["targets"]["all"]
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