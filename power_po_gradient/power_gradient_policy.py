import lasagne.layers as L
import lasagne.nonlinearities as NL

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
            neat_output_dim=20,
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

        if neat_network is None:
            neat_network = MLP(
                input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs,),
                output_dim=neat_output_dim,
                hidden_sizes=(4,),
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.identity,
            )

        if prob_network is None:
            prob_network = MLP(
                input_shape=(L.get_output_shape(neat_network.output_layer)[1],),
                output_dim=env_spec.action_space.n,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
            )

        self._phi = neat_network.output_layer
        self._obs = neat_network.input_layer
        self._neat_output = ext.compile_function([neat_network.input_layer.input_var], L.get_output(neat_network.output_layer))

        self.prob_network = prob_network
        self._l_prob = prob_network.output_layer
        self._l_obs = prob_network.input_layer
        self._f_prob = ext.compile_function([prob_network.input_layer.input_var], L.get_output(prob_network.output_layer))

        self._dist = Categorical(env_spec.action_space.n)

        super(PowerGradientPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [neat_network.output_layer])
        LasagnePowered.__init__(self, [prob_network.output_layer])

    def action_log_prob_sym(self, obs_var, act_var):
        """
        Aaron: obtain the symbolic representation for the log probability of taking a specific action in the form [0, 1, 0, ...]
        :param obs_var: 
        :param act_var: 
        :return: 
        """
        # phi = L.get_output(self._phi, {self._obs: obs_var})
        prob = L.get_output(self._l_prob, {self._l_obs: obs_var})
        return T.log(T.dot(prob, act_var).mean() + TINY)

    @overrides
    def dist_info_sym(self, obs_var):
        # phi = L.get_output(self._phi, {self._obs: obs_var})
        return dict(prob=L.get_output(self._l_prob, {self._l_obs: obs_var}))

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        # phi = self._neat_output([flat_obs])[0]
        prob = self._f_prob([flat_obs])[0]  # does phi need to be flattened
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        # phis = self._neat_output(flat_obs)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    def get_cumulative_log_probability(self, observations, actions):
        return np.sum(np.log((self.get_actions(observations)[1]['prob'] * actions).sum(axis=1) + TINY))

    @property
    def distribution(self):
        return self._dist

    def get_output_layer(self):
        layers = self.prob_network.layers
        return layers[-1].get_params(trainable=True)[0].get_value()

    def save_policy(self, identifier):
        np.savez('model-{0}.npz'.format(identifier), L.get_all_param_values(self.prob_network))

    def load_policy(self, filename='model.npz'):
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        L.set_all_param_values(self.prob_network, param_values[0])

