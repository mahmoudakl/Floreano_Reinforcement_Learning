import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(individual_dir):
	"""

	"""
	trajectory = np.load(individual_dir + '/trajectory.npy')
	fig = plt.figure()
	plt.gca()
	plt.xticks([], [])
	plt.yticks([], [])
	plt.ylim(-3, 3)
	plt.xlim(-3.9, 3.9)
	x_axis = [x[0] for x in trajectory]
	y_axis = [y[1] for y in trajectory]
	plt.plot([float(x) for x in x_axis], [float(y) for y in y_axis])
	fig.savefig(individual_dir + '/trajectory.png')
	plt.close()


def plot_wheel_speeds(individual_dir):
	"""
	Reads the saved wheel speeds and saves a plotted figure
	"""

	wheel_speeds = np.load(individual_dir + '/wheel_speeds.npy')
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


def plot_results(generation_dir):
	"""
	"""
	individuals = [s for s in os.listdir(generation_dir) if 'individual' in s]
	for i in individuals:
		print i
		plot_spikes(generation_dir + '/' + i)
		plot_wheel_speeds(generation_dir + '/' + i)
		plot_trajectory(generation_dir + '/' + i)
