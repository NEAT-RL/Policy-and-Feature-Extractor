#!/usr/bin/env python

"""Description:
NEAT agent solves the flappybird game problem
Created by Yiming
"""

#!/usr/bin/env python
from rllab.envs.flappy_bird_env import FlappyBirdEnv
from feature_extractor.power_gradient_policy import PowerGradientPolicy
from rllab.envs.normalized_env import normalize

"""Description:
NEAT works on GYM Cart Pole
Workable version for both  CartPole and MountainCar
"""
import pickle
import datetime
import dateutil.tz
import neat
import numpy as np
#hyper paramerters
num_of_steps = 100000
num_of_episodes = 1
num_of_generations = 201
is_render = False

config_path = 'config/flappybird'
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
env = FlappyBirdEnv()

policy = PowerGradientPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)
# Load policy parameters = weights + bias of pretrained network
policy.load_policy('policy_parameters/model-flappybird.npz')

def evaluation(genomes, config):
    global steps_to_train, generation
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness, steps = do_rollout(net, is_render)
        steps_to_train = steps_to_train + steps

    # sort the genomes by fitness
    gid, best_genome = max(genomes, key=lambda x: x[1].fitness)

    if generation % 20 == 0:
        with open('best_genomes/flappybird-gen-{0}-fitness-{1}-genome'.format(generation, best_genome.fitness), 'wb') as f:
            pickle.dump(best_genome, f)

    generation = generation + 1


def do_rollout(agent, render=False):
    rewards = []
    for i in range(10):
        ob = env.reset()
        t = 0
        total_rewards = 0
        for t in range(num_of_steps):
            outputs = agent.activate(ob)
            action, prob = policy.get_action(outputs)
            (ob, reward, done, _info) = env.step(action)
            total_rewards += reward
            if render and t % 3 == 0:
                env.render()
            if done:
                break

        rewards.append(total_rewards)

    return np.mean(rewards)

def run(config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    winner = p.run(evaluation, num_of_generations)

    with open('best_genomes/flappybird-gen-winner-fitness-{0}-genome'.format(winner.fitness), 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    steps_to_train = 0
    generation = 0
    run(config=config_path)
