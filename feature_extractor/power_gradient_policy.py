import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy
from rllab.spaces import Discrete
import numpy as np

import theano.tensor as T

TINY = 1e-8


class PowerGradientPolicy(StochasticPolicy, LasagnePowered):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(),
            hidden_nonlinearity=NL.tanh,
            num_seq_inputs=1,
            neat_output_dim=(20, ),
            neat_network=None,
            prob_network=None,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        if prob_network is None:
            prob_network = MLP(
                input_shape=neat_output_dim,
                output_dim=env_spec.action_space.n,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
            )
        self.neat_network = neat_network
        self.neat_output_dim = neat_output_dim
        self.prob_network = prob_network
        self._l_prob = prob_network.output_layer
        self._l_obs = prob_network.input_layer
        self._f_prob = ext.compile_function([prob_network.input_layer.input_var], L.get_output(prob_network.output_layer))

        self._dist = Categorical(env_spec.action_space.n)

        super(PowerGradientPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [prob_network.output_layer])

    @overrides
    def dist_info(self, obs, state_infos=None):
        neat_output_var = self.get_neat_output(obs)
        return dict(prob=self._f_prob(neat_output_var))

    @overrides
    def get_action(self, observation):
        neat_output = self.neat_network.activate(observation)
        flat_obs = self.observation_space.flatten(neat_output)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        neat_output_vars = self.get_neat_output(observations)
        flat_obs = self.observation_space.flatten_n(neat_output_vars)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist

    def get_neat_output(self, obs_var):
        if len(obs_var[0]) == self.input_shape[0]:
            return obs_var
        else:
            neat_output_var = []
            for obs in obs_var:
                neat_output_var.append(self.neat_network.activate(obs))

            neat_output_var = np.array(neat_output_var)
            return neat_output_var

    def get_output_layer(self):
        layers = self.prob_network.layers
        return layers[-1].get_params(trainable=True)[0].get_value()

    def save_policy(self, identifier):
        np.savez('model-{0}.npz'.format(identifier), L.get_all_param_values(self.prob_network))

    def load_policy(self, filename='model.npz'):
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        L.set_all_param_values(self.prob_network, param_values[0])

