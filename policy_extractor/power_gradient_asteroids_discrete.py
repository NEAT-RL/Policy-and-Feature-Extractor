from policy_extractor.power_gradient import POWERGradient
from policy_extractor.power_gradient_policy import PowerGradientPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

import rllab.misc.logger as logger
import os.path as osp
import datetime
import dateutil.tz
import uuid
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learning_rate', nargs='?',
                        default='0.000001',
                        help='Select the learning rate to use')

    args = parser.parse_args()

    """
    Setup Logging of data into csv files.
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'asteroids/power_po_gradient/experiment_%s_%s' % (timestamp, rand_id)
    default_log_dir = osp.abspath(osp.join(osp.dirname(__file__), '..')) + "/data"
    log_dir = osp.join(default_log_dir, default_exp_name)
    tabular_log_file = osp.join(log_dir, "progress.csv")
    logger.add_tabular_output(tabular_log_file)

    env = normalize(normalize(GymEnv("Asteroids-ram-v0")))

    policy = PowerGradientPolicy(
        env_spec=env.spec,
        neat_output_dim=64,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = POWERGradient(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100000,
        max_path_length=10000,
        n_itr=200,
        discount=0.99,
        step_size=float(args.learning_rate),
    )
    algo.train()

    policy.save_policy('asteroids_{0}'.format(algo.average_return()))
