# Policy and Feature Extractor

This repo contains two main sections: [**Policy Extractor**](https://github.com/NEAT-RL/Policy-and-Feature-Extractor/tree/master/policy_extractor) and [**Feature Extractor**](https://github.com/NEAT-RL/Policy-and-Feature-Extractor/tree/master/feature_extractor).

## 1. Policy Extractor
Policy Extractor is used to for extracting _temporary_ policy parameters to be used for extractoring extracting NEAT Feature parameters. 

The Policy consists of a NEAT MLP network and Policy MLP. The Policy first extracts high level state features using the NEAT MLP, then uses the output from the NEAT MLP as input to Policy MLP to predict actions for a game.

The core idea is to train the algorithm to extract useful policy parameter so that we can use NEAT to extract useful state features.

Once the algorithm has been trained. The `policy.save_policy` method can be used to save the policy MLP weights and bias for use in the Feature Extraction code. An example of this is `policy.save_policy('asteroids_{0}'.format(algo.average_return()))` which saves the policy parameters with the name `asteriods_[average return of algorithm].npz`.

### NEAT MLP
The input to the NEAT MLP is the environment observation.  
The NEAT MLP has random weights and 12x12 hiden layers. The output of the NEAT MLP is can be set to any value using the _neat_output_dim_ parameter, by default it is set to 20. You can optionally pass an existing NEAT MLP by using the _neat_network_ parameter. 

### Policy MLP
The input to the Policy MLP is the output from NEAT MLP.  
The hiden layers can be set using the _hidden_sizes_ parameter. The output of the policy MLP represent the environment action space.


## 2. Feature Extractor

Feature Extractor is used for extracting useful NEAT Network parameters. The idea is to use the _temporary_ policy parameters from the Policy Extractor method to evolve NEAT networks and save the NEAT parameters.

> N.B. Feature Extraction on complex Atari games can take a very long time. If you are using grid-solar, your request may be terminated by overusing the system resources.   
> It is recommended you run the Feature Extractor on a multicore system with many threads. I found that running feature extractor for 40-60 generation is enough.

### Policy Parameters
A `PowerGradientPolicy` policy object is used for creating a policy. The policy's _neat_output_dim_ paramter needs to be defined. The _hidden_sizes_ parameter also needs to be defined. An example for the asteroids game is shown below:
```Python
policy = PowerGradientPolicy(
        env_spec=env.spec,
        neat_output_dim=(64, ),
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 32)
)
```

The next thing that needs to be done is to load saved policy parameters for the given environment. This can be done using the `policy.load_policy() method`  e.g. `policy.load_policy('policy_parameters/model-asteroids.npz')`.

### Fitness Function
Because we use a stochastic policy, we use the average total reward of 10 rollouts as the fitness value of a NEAT Network.

### NEAT Network 
The best NEAT network (based on the highest fitness value) every 20 generations should be saved e.g. 
```Python
if generation % 20 == 0:
        with open('best_genomes/asteroids-gen-{0}-fitness-{1}-genome'.format(generation, best_genome.fitness), 'wb') as f:
            pickle.dump(best_genome, f)

```

The winner NEAT network is also saved. e.g.

```Python
winner = p.run(evaluation, num_of_generations)

with open('best_genomes/asteroids-gen-winner-fitness-{0}-genome'.format(winner.fitness), 'wb') as f:
    pickle.dump(winner, f)
```
