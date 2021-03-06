from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging

logger = logging.getLogger(__name__)

dna = np.array([int(x) for x in '%s'.split(',')]).reshape(10, 29)

receptors = []
for r in range(1, 19):
    receptors.append(np.nonzero(dna[:, r])[0])


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

    row_counter = 0
    for row in dna:
        logger.info(row)
        n = np.array(row)
        r_type = 'excitatory'
        if n[0] == 0:
            r_type = 'inhibitory'
        for i in range(19, 29):
            if n[i] == 1:
                sim.Projection(presynaptic_population=CIRCUIT[row_counter:1 + row_counter],
                               postsynaptic_population=CIRCUIT[i - 19:i - 18],
                               connector=sim.OneToOneConnector(), synapse_type=SYN,
                               receptor_type=r_type)

        row_counter += 1

    sim.initialize(population, v=population.get('v_rest'))

    logger.debug("Circuit description: " + str(population.describe()))

    return population


circuit = create_brain()
