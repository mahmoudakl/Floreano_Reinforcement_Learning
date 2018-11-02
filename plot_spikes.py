import csv
from matplotlib import pyplot as plt


def plot_spikes(spikes_file):
	with open(spikes_file, 'rb') as csv_file:
	    spike_reader = csv.reader(csv_file)
	    spikes = list(spike_reader)[1:]

	for i in range(len(spikes)):
	    spikes[i][0] = int(spikes[i][0])
	    spikes[i][1] = float(spikes[i][1])

	neurons = column(spikes, 0)
	times = column(spikes, 1)

	plt.gca()
	plt.ylim(2, 13)
	ply.yticks(3, 12)
	plt.xlabel('Time [ms]')
	plt.ylabel('Neuron ID')
	plt.plot(times, neurons, 'bo')
	plt.show()


def column(matrix, i):
    return [row[i] for row in matrix]
