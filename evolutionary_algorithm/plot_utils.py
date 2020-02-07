import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_average_fitness(results_dir):
    """
    Plots the average fitness per generation across all generations and saves the plot in the
    results directory

    :param results_dir: The directory containing all generation results as subdirectories
    """
    avg_fitness = []
    generations = [d for d in os.listdir(results_dir) if 'generation' in d]
    for g in generations:
        avg_fitness.append(np.load(results_dir + '/' + g + '/average_fitness.npy', allow_pickle=True))

    fig = plt.figure()
    plt.plot(avg_fitness)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness per Generation')
    fig.savefig(results_dir + '/avg_fitness.png')
    plt.close()


def plot_sensory_input(pixel_values, ylabel="Pixel intensities"):
    """
    Plots array values in bars. Useful for viewing input pixel intensities or receptors' firing
    probabilities

    :param pixel_values: Array containing values to be plotted in bars
    :param ylabel: Label to be displayed on the y-axis of the plot
    """
    plt.bar(range(len(pixel_values)), pixel_values, color='black')
    plt.xlim(0, len(pixel_values))
    plt.ylim(0, np.abs(max(pixel_values)))
    plt.xlabel("Receptors")
    plt.ylabel(ylabel)
    plt.show()


def plot_best_fitness(results_dir):
    """
    Plots the best fitness value achieved per generation across all generations and saves the plot
    in the results directory

    :param results_dir: The directory containing all generation results as subdirectories
    """
    best_fitness = []
    generations = [d for d in os.listdir(results_dir) if 'generation' in d]
    for g in generations:
        best_performer = np.load(results_dir + '/' + g + '/top_performers_indices.npy', allow_pickle=True)[0]
        best_fitness.append(np.load(results_dir + '/' + g + '/individual_{}/fitness_value.npy'.
                                    format(best_performer), allow_pickle=True))

    fig = plt.figure()
    plt.plot(best_fitness)
    plt.xlabel('Generations')
    plt.ylabel('Best achieved fitness value')
    fig.savefig(results_dir + '/best_fitness.png')
    plt.close()


def plot_robot_path(individual_dir):
    """

    """
    robot_path = np.load(individual_dir + '/robot_path.npy', allow_pickle=True)
    robot_initial_pose = np.load(individual_dir + '/initial_pose.npy', allow_pickle=True)
    fig = plt.figure()
    plt.gca()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylim(-3, 3)
    plt.xlim(-3.9, 3.9)
    x_axis = [x[0] for x in robot_path]
    y_axis = [y[1] for y in robot_path]
    plt.plot([float(x) for x in x_axis], [float(y) for y in y_axis])
    plt.plot(float(x_axis[0]), float(y_axis[0]), marker=(3, 0, np.rad2deg(robot_initial_pose[-1])))
    fig.savefig(individual_dir + '/robot_path.png')
    plt.close()


def plot_wheel_speeds(individual_dir):
    """
    Reads the saved wheel speeds and saves a plotted figure
    """

    wheel_speeds = np.load(individual_dir + '/wheel_speeds.npy', allow_pickle=True)
    left_wheel = [float(t[1]) for t in wheel_speeds]
    right_wheel = [float(t[2]) for t in wheel_speeds]
    fig = plt.figure()
    plt.plot(range(len(left_wheel)), left_wheel, 'b', label='Left Wheel')
    plt.plot(range(len(right_wheel)), right_wheel, 'r', label='Right Wheel')
    plt.ylim(-1, 1)
    plt.xlabel('Times [ms]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    fig.savefig(individual_dir + '/wheel_speeds.png')
    plt.close()


def plot_spikes(individual_dir):
    """
    Read the all_spikes csv file and plot the spikes emitted by each neuron
    """

    file_path = individual_dir + '/all_spikes.csv'
    with open(file_path, 'rb') as csv_file:
        spike_reader = csv.reader(csv_file)
        spikes = list(spike_reader)[1:]

    for i in range(len(spikes)):
        spikes[i][0] = int(float(spikes[i][0]))
        spikes[i][1] = float(spikes[i][1])

    spikes = [s for s in spikes if s[1] <= 40000]

    neurons = column(spikes, 0)
    times = column(spikes, 1)
    fig = plt.figure()
    plt.gca()
    plt.ylim(2, 13)
    plt.yticks(range(3, 14), range(1, 11))
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron ID')
    plt.plot(times, neurons, 'bo')
    fig.savefig(individual_dir + '/spikes.png')
    plt.close()


def column(matrix, i):
    """
    helper function to return a column from a 2d array
    """
    return [row[i] for row in matrix]


def plot_results(results_dir):
    """
    """
    generations_dirs = [g for g in os.listdir(results_dir) if 'generation' in g]
    for generation in generations_dirs:
        generation_dir = results_dir + '/' + generation
        individuals = [s for s in os.listdir(generation_dir) if 'individual' in s]
        for i in individuals:
            print "{}\n{}\n==============".format(generation, i)
            plot_spikes(generation_dir + '/' + i)
            plot_wheel_speeds(generation_dir + '/' + i)
            plot_robot_path(generation_dir + '/' + i)
