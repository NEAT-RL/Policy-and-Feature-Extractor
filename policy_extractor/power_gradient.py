from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import theano

import numpy as np
import random

TINY = 1e-3

class POWERGradient(BatchPolopt):
    """
    The POWER policy gradient based algorithm (use calculated policy gradient for direct policy search).
    """

    numSampledPaths = 10

    def __init__(
            self,
            step_size=0.0001,
            **kwargs
    ):
        self.step_size = step_size
        super(POWERGradient, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        obv_var = self.env.observation_space.new_tensor_variable(
            'obv',
            extra_dims=0,
        )
        act_var = self.env.action_space.new_tensor_variable(
            'act',
            extra_dims=0,
        )

        self.policyParametes = self.policy.get_params(trainable=True)
        po_log_grad = theano.grad(self.policy.action_log_prob_sym(obv_var, act_var), self.policyParametes, disconnected_inputs='ignore')
        flat_pol_log_grad = ext.flatten_tensor_variables(po_log_grad)
        self.polLogGradFunc = theano.function(
            inputs=[obv_var, act_var],
            outputs=flat_pol_log_grad,
            allow_input_downcast=True
        )

        # Add gS to support RMSProp.
        self.gS = np.zeros(len(self.policy.get_param_values()))

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data, paths):
        sortedPaths = paths

        # select a subset of paths for training.
        if len(sortedPaths) > POWERGradient.numSampledPaths:
            # select a random subset of paths for tranining.
            # selected_samples = random.sample(range(len(sortedPaths)), 10)
            # sortedPaths = [sortedPaths[x] for x in selected_samples]

            # select the subset of best paths for training.
            sortedPaths = sorted(paths, key=lambda path: np.sum(path["rewards"]), reverse=True)
            sortedPaths = sortedPaths[0:POWERGradient.numSampledPaths]

        processed_samples = self.sampler.process_samples(itr, sortedPaths)

        all_input_values = ext.extract(
            processed_samples,
            "observations", "actions", "path_rewards"
        )

        polGrad = np.array([0.] * len(self.policy.get_param_values()))
        for obv, act, path_rew in zip(*all_input_values):
            polGrad = polGrad + path_rew * np.array(self.polLogGradFunc(obv, act))
        polGrad = polGrad / POWERGradient.numSampledPaths

        # RMSProp update of policy parameters.
        self.gS = 0.9 * self.gS + 0.1 * (polGrad ** 2)
        newPolParams = self.policy.get_param_values() + self.step_size * polGrad / np.sqrt(self.gS + TINY)
        self.policy.set_param_values(newPolParams)

        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
