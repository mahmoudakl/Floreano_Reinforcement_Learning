# -*- coding: utf-8 -*-
"""
This file contains the setup of the neuronal network used in the Floreano experiment
"""
# pragma: no cover


from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging

logger = logging.getLogger(__name__)

l = np.random.randint(2, size=(10, 29))
receptors = []
for r in range(11, 29):
    receptors.append(np.nonzero(l[:, r])[0])
"""
Structure of the dna array: 
-Binary values
-Each row represents a neuron
-First bit: the neuron is inhibitory (0) or excitatory (1) 
-Next 10 bits: from which brain neurons does the current neuron receive input (1), self connections allowed
-Next 18 bits: which sensory neurons are connected (1) or disconnected (0) to the neuron (only exitatory synapses)
"""


def create_brain(dna=l):
    """
    Initializes PyNN with the neuronal network that has to be simulated
    """

    NEURONPARAMS = {'v_rest': -60.5,
                    'tau_m': 4.0,
                    'tau_refrac': 2.0,
                    'tau_syn_E': 10.0,
                    'tau_syn_I': 10.0,
                    'v_thresh': -60.4,
                    'v_reset': -60.5}

    SYNAPSE_PARAMS = {"weight": 1.0,
                      "delay": 2.0}

    population = sim.Population(10, sim.IF_cond_alpha())
    population[0:10].set(**NEURONPARAMS)

    # Connect neurons
    CIRCUIT = population

    SYN = sim.StaticSynapse(**SYNAPSE_PARAMS)

    row_counter = 0
    for row in dna:
        logger.info(row)
        n = np.array(row)
        for i in range(1, 11):
            if n[i] == 1:
                r_type = 'excitatory' if dna[i - 1][0] else 'inhibitory'
                logger.info('Synapse from Neuron: ' + str(i) + ' to ' + str(row_counter + 1)+' ' +
                            r_type)
                sim.Projection(presynaptic_population=CIRCUIT[row_counter:row_counter + 1],
                               postsynaptic_population=CIRCUIT[i-1:i],
                               connector=sim.OneToOneConnector(), synapse_type=SYN,
                               receptor_type=r_type)

        row_counter += 1

    sim.initialize(population, v=population.get('v_rest'))

    logger.debug("Circuit description: " + str(population.describe()))

    return population


circuit = create_brain()
