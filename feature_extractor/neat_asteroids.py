import neat
import numpy as np
import pickle
import multiprocessing
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from feature_extractor.power_gradient_policy import PowerGradientPolicy
import datetime
import dateutil.tz
import os.path as osp

PROJECT_PATH = osp.abspath(osp.dirname(__file__))

#hyper paramerters
num_of_generations = 201
num_of_steps = 10000

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

is_render = False

# config file
config_path = 'config/asteroids'
env = normalize(normalize(GymEnv("Asteroids-ram-v0")))


policy = PowerGradientPolicy(
        env_spec=env.spec,
        neat_output_dim=64,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 32)
)

policy.load_policy('policy_parameters/model-asteroids.npz')

def do_rollout(agent, render=False):
    total_reward = 0
    ob = env.reset()
    t = 0
    for t in range(num_of_steps):
        outputs = agent.activate(ob)
        action, prob = policy.get_action(outputs)
        (ob, reward, done, _info) = env.step(action)
        total_reward += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_reward, t


def eval_genome(genome, config):
    """
    This function will be run in threads by ThreadedEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = do_rollout(net, is_render)
    # if fitness >= 200:
        # logger.debug("Found solution so terminating algorithm")
        # set fitness to very high to terminate NEAT
        # fitness = 1000
    return fitness


def evaluation(genomes, config):
    global generation, steps_to_train
    nets = []
    for gid, g in genomes:
        nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

    if pool is None:
        for genome, net in nets:
            genome.fitness, steps = do_rollout(net, is_render)
            steps_to_train += steps
    else:
        jobs = []
        for genome, net in nets:
            jobs.append(pool.apply_async(do_rollout, (net,)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness, steps = job.get(timeout=None)
            steps_to_train += steps

    # sort the genomes by fitness
    gid, best_genome = max(genomes, key=lambda x: x[1].fitness)

    if generation % 20 == 0:
        with open('best_genomes/asteroids-gen-{0}-fitness-{1}-genome'.format(generation, best_genome.fitness), 'wb') as f:
            pickle.dump(best_genome, f)

    generation = generation + 1


def run(config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    # Run for up to 300 generations.
    # pe = neat.ThreadedEvaluator(4, eval_genome)
    # winner = p.run(pe.evaluate, 100)
    winner = p.run(evaluation, num_of_generations)

    with open('best_genomes/asteroids-gen-winner-fitness-{0}-genome'.format(winner.fitness), 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    generation = 0
    steps_to_train = 0
    pool = None  # if multiprocessing.cpu_count() < 2 else multiprocessing.Pool()
    run(config=config_path)
