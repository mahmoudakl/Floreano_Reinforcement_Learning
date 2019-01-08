# This file contains helper functions to support running the Floreano Experiment

import numpy as np


def get_wheel_speeds(individual_dir, corrected=False):
    """
    Loads the saved wheel speeds data for a specific individual. The loaded data can either be the
    raw data recorded from the experiment, or the processed data that corrects for collisions

    :param individual_dir: The directory containing the individual's saved results
    :param corrected: Flag indicating whether the data loaded should be the raw data, or the
                      processed data that corrects for collisions
    """
    if not corrected:
        file_path = individual_dir + '/wheel_speeds.csv'
    else:
        file_path = individual_dir + '/corrected_wheel_speeds.csv'
    wheel_speeds = [i.strip().split(',') for i in open(file_path).readlines()][1:401]
    np.save(individual_dir + '/wheel_speeds', wheel_speeds)
    return wheel_speeds


def get_trajectory(individual_dir):
    """

    :param individual_dir: The directory containing the individual's saved results
    """
    file_path = individual_dir + '/robot_position.csv'
    trajectory = [i.strip().split(',') for i in open(file_path).readlines()][1:401]
    np.save(individual_dir + '/trajectory', trajectory)
    return trajectory


def fitness_function(wheel_speeds):
    """
    Calculates the fitness function described in the Floreano paper
    according to the saved wheel speeds
    """
    left_wheel = [float(t[1]) for t in wheel_speeds]
    right_wheel = [float(t[2]) for t in wheel_speeds]
    fitness = 0
    for i in range(len(left_wheel)):
        if left_wheel[i] > 0 and right_wheel[i] > 0:
            fitness += (left_wheel[i] + right_wheel[i])
    return fitness/float(2*len(left_wheel))


def get_top_performers(generation_dir, num_performers=15):
    """
    Extract the indices of the top individuals from the fitness log

    :param generation_dir: Directory where all generation results are stored
    :param num_performers: number for top performers to look for. Default value
                           is 15, which corresponds to a truncation threshold of
                           25% in the original experiment
    """
    fitness_log = []
    top_performers_indices = []
    top_performers_individuals = []

    # store all fitness values in a list
    for i in range(60):
        fitness_log.append(np.load(generation_dir + '/individual_{}'.format(i)
                                   + '/fitness_value.npy'))

    # save generation average fitness value
    avg_fitness = np.average(fitness_log)
    np.save(generation_dir + '/average_fitness', avg_fitness)

    # find the top performing individuals
    for i in range(num_performers):
        max_index = np.argmax(fitness_log)
        top_performers_indices.append(max_index)
        top_performers_individuals.append(np.load(generation_dir +
                                                  '/individual_{}'.format(max_index) +
                                                  '/genetic_string.npy'))
        fitness_log[max_index] = -1

    np.save(generation_dir + '/top_performers_indices', top_performers_indices)
    np.save(generation_dir + '/top_performers_strings',
            top_performers_individuals)


def one_point_crossover(parent1, parent2):
    """
    Performs one-point cross over on two genetic strings at a random point and
    returns two new strings

    :param parent1: The first string
    :param parent2: The second string
    """
    parent1 = parent1.reshape(290)
    parent2 = parent2.reshape(290)
    child1 = np.zeros(290, dtype=int)
    child2 = np.zeros(290, dtype=int)
    point = np.random.randint(len(parent1))
    for i in range(point):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    for i in range(point, 290):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
    child1 = child1.reshape(10, 29)
    child2 = child2.reshape(10, 29)
    return child1, child2


def bit_mutation(population):
    """
    Performs bit mutation on every item of every binary genetic string in a
    population with 5% probability

    :param population: A list of binary genetic strings
    """
    for individual in population:
        individual = individual.reshape(290)
        for i in range(290):
            if np.random.rand() < 0.05:
                individual[i] = 0 if individual[i] else 1
    return population


def get_unique_pairs(population):
    """
    Returns all unique pairs in a list of items as a list of tuples

    :param population: List of items
    """
    pairs = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            pairs.append((i, j))
    return pairs


def evolve_new_generation(generation_dir):
    """
    Evolve a new generation based on the top performers of a previous generation
    using bt mutation, one-point cross over and elitism

    :param generation_dir: The directory where all generation results are stored
    """
    top_performers = np.load(generation_dir + '/top_performers_strings.npy')
    population = []
    for i in range(len(top_performers)):
        for j in range(4):
            population.append(top_performers[i])
    pairs = get_unique_pairs(population)
    for i in pairs:
        if np.random.rand() < 0.1:
            parent1 = population[i[0]]
            parent2 = population[i[1]]
            child1, child2 = one_point_crossover(parent1, parent2)
            population[i[0]] = child1
            population[i[1]] = child2

    population = bit_mutation(population)
    rand = np.random.randint(len(population))
    population[rand] = top_performers[0]

    f = open("populations.txt", "w")
    f.write(str(population))
    f.write("\n")
    f.close()

    return population


def correct_for_collisions(individual_dir):
    """
    Loops through the robot trajectory and sets wheel speeds to zeros if the robot exceeded certain
    thresholds in the x- and y-positions. These thresholds indicate that the robot has hit the wall

    :param individual_dir: The directory containing the individual's saved results
    """

    # convert wheel_speeds to a numpy array to be able to broadcast zeros if a collision is detected
    wheel_speeds = np.asarray(get_wheel_speeds(individual_dir))
    trajectory = get_trajectory(individual_dir)
    x_axis = [x[0] for x in trajectory]
    y_axis = [y[1] for y in trajectory]
    collision = False

    for i in range(len(wheel_speeds)):
        if float(y_axis[i]) >= 2.4 or float(y_axis[i]) <= -2.4:
            wheel_speeds[i:, 1] = '0.0'
            wheel_speeds[i:, 2] = '0.0'
            collision = True
        elif float(x_axis[i]) >= 3 or float(x_axis[i]) <= -3:
            wheel_speeds[i:, 1] = '0.0'
            wheel_speeds[i:, 2] = '0.0'
            collision = True

        if collision:
            break

    # save corrected wheel speeds
    np.save(individual_dir + '/corrected_wheel_speeds', wheel_speeds)
    return collision


brain = """from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging

logger = logging.getLogger(__name__)

dna = np.array([int(x) for x in '%s'.split(',')]).reshape(10, 29)

receptors = []
for r in range(1,19):
    receptors.append(np.nonzero(dna[:,r])[0])


def create_brain():

    NEURONPARAMS = {'v_rest': -60.5,
                    'tau_m': 4.0,
                    'tau_refrac': 2.0,
                    'tau_syn_E': 10.0,
                    'tau_syn_I': 10.0,
                    'e_rev_E': 0.0,
                    'e_rev_I': -75.0,
                    'v_thresh': -60.4,
                    'v_reset': -60.5}

    SYNAPSE_PARAMS = {"weight": 1.0,
                      "delay": 2.0}

    population = sim.Population(10, sim.IF_cond_alpha())
    population[0:10].set(**NEURONPARAMS)


    # Connect neurons
    CIRCUIT = population

    SYN = sim.StaticSynapse(**SYNAPSE_PARAMS)

    row_counter=0
    for row in dna:
        logger.info(row)
        n = np.array(row)
        r_type = 'excitatory'
        if n[0]==0:
            r_type = 'inhibitory'
        for i in range(19,29):
            if n[i]==1:
                sim.Projection(presynaptic_population=CIRCUIT[row_counter:1+row_counter], postsynaptic_population=CIRCUIT[i-19:i-18], connector=sim.OneToOneConnector(), synapse_type=SYN, receptor_type=r_type)
        
        row_counter+=1

    sim.initialize(population, v=population.get('v_rest'))

    logger.debug("Circuit description: " + str(population.describe()))

    return population


circuit = create_brain()
"""