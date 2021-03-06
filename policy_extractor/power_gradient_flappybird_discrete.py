import argparse
import datetime
import os.path as osp
import uuid

import dateutil.tz
import rllab.misc.logger as logger
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.flappy_bird_env import FlappyBirdEnv
from rllab.envs.normalized_env import normalize

from policy_extractor.power_gradient import POWERGradient
from policy_extractor.power_gradient_policy import PowerGradientPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learning_rate', nargs='?',
                        default='0.001',
                        help='Select the learning rate to use')

    args = parser.parse_args()

    """
    Setup Logging of data into csv files.
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'flappybird/power_po_gradient/experiment_%s_%s' % (timestamp, rand_id)
    default_log_dir = osp.abspath(osp.join(osp.dirname(__file__), '..')) + "/data"
    log_dir = osp.join(default_log_dir, default_exp_name)
    env = normalize(normalize(FlappyBirdEnv()))

    tabular_log_file = osp.join(log_dir, "progress.csv")
    logger.add_tabular_output(tabular_log_file)

    policy = PowerGradientPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = POWERGradient(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=500000,
        max_path_length=100000,
        n_itr=100,
        discount=0.99,
        step_size=float(args.learning_rate),
    )
    algo.train()

    policy.save_policy('flappybird_{0}'.format(algo.average_return()))
