from rllab.envs.flappy_bird_env import FlappyBirdEnv
from power_po_gradient.power_gradient import POWERGradient
from power_po_gradient.power_gradient_policy import PowerGradientPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize

from rllab import config
import rllab.misc.logger as logger
import os.path as osp
import datetime
import dateutil.tz
import uuid

"""
Setup Logging of data into csv files.
"""
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
LOG_DIR = PROJECT_PATH + "/data"

now = datetime.datetime.now(dateutil.tz.tzlocal())

# avoid name clashes when running distributed jobs
rand_id = str(uuid.uuid4())[:5]
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

default_exp_name = 'flappybird/power_po_gradient/experiment_%s_%s' % (timestamp, rand_id)

log_dir = osp.join(LOG_DIR, default_exp_name)
env = normalize(normalize(FlappyBirdEnv()))

tabular_log_file = osp.join(log_dir, "progress.csv")
logger.add_tabular_output(tabular_log_file)

policy = PowerGradientPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(6,)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = POWERGradient(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=1000,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
)
algo.train()
