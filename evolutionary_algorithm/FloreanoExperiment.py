import os
import sys
import time
import rospy
import shutil
import logging
import evolution_utils

import numpy as np

from gazebo_msgs.msg import ModelState
from IPython.display import clear_output
from tf.transformations import quaternion_from_euler
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach

# disable global logging from the virtual coach
logging.disable(logging.INFO)
logging.getLogger('rospy').propagate = False
logging.getLogger('rosout').propagate = False

vc = VirtualCoach(environment='local', storage_username='nrpuser', storage_password='password')

display_episode_tf = """@nrp.Robot2Neuron()
def display_episode_number(t):
    clientLogger.advertise('%s')
"""


class FloreanoExperiment(object):

    def __init__(self, experiment_name, population, generations):
        self.experiment_name = experiment_name
        self.experiment_dir = os.environ['HOME'] + '/.opt/nrpStorage/' + self.experiment_name
        self.last_status = [None]
        self.sim = None
        self.initial_pose = np.array([])
        self.generations = generations

        self._set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState)

        # keep track of the simulation time, as the reset call is not executed at the precise time
        self.cur_sim_time = 0

        # remove any previously saved csv_recorders' data
        csv_dirs = [s for s in os.listdir(self.experiment_dir) if "csv_records" in s]
        for directory in csv_dirs:
            shutil.rmtree(self.experiment_dir + '/' + directory)

        # Check if evolutionary experiment has been run before
        # If yes, load previously evolved population and continue
        previous_generations = [int(s.split('_')[1]) for s in os.listdir(self.experiment_dir)
                                if "generation" in s]
        previous_generations.sort()
        if len(previous_generations) == 0:
            self.population = population
            self.cur_gen = 0
        else:
            # current generation number
            self.cur_gen = previous_generations[-1]
            last_gen_dir = self.experiment_dir + '/generation_{}'.format(self.cur_gen)
            individuals = [s for s in os.listdir(last_gen_dir) if "individual" in s]
            # If generation has not been fully simulated, delete it and start from the previous one
            if len(individuals) >= 60:
                self.population = evolution_utils.evolve_new_generation(last_gen_dir)
                self.cur_gen += 1
            else:
                shutil.rmtree(last_gen_dir)
                # If there's only one incomplete generation, start over
                if self.cur_gen == 0:
                    self.population = population
                else:
                    last_gen_dir = self.experiment_dir + '/generation_{}'.format((self.cur_gen - 1))
                    self.population = evolution_utils.evolve_new_generation(last_gen_dir)

    def set_random_robot_pose(self):
        """
        sets the robot pose to a random pose within the box
        """
        msg = ModelState()
        msg.model_name = 'robot'
        msg.reference_frame = 'world'
        msg.scale.x = msg.scale.y = msg.scale.z = 1.0

        initial_pose = np.array([np.random.uniform(-3, 3), np.random.uniform(-2.4, 2.4), 0.18,
                                 np.random.uniform(0, np.pi)])

        # random position within box
        msg.pose.position.x = initial_pose[0]
        msg.pose.position.y = initial_pose[1]
        msg.pose.position.z = initial_pose[2]

        # random orientation
        quaternion = quaternion_from_euler(0, 0, initial_pose[3])
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        # publish message on ros topic
        self._set_model_state.publish(msg)

        return initial_pose

    def wait_condition(self, timeout, condition, verbose=False):
        """
        Helper method that blocks for the timeout specified, until the condition
        given has been fulfilled and continuously prints the current simulation time
        """
        start = time.time()
        while time.time() < start + timeout:
            time.sleep(1)
            if verbose:
                sys.stdout.write("Simulation Time: {}\r".format(
                    self.last_status[0]['simulationTime']))
                sys.stdout.flush()
            if self.last_status[0] is not None:
                if condition(self.last_status[0]):
                    return
        raise Exception('Condition check failed')

    def on_status(self, status):
        """Prepends the most recent ROS status message to the last_status array"""
        self.last_status[0] = status

    def save_simulation_data(self, generation, trial):
        """
        Saves the simulation csv data to the respective individual's directory inside the
        experiment's directory
        """
        csv_dirs = [s for s in os.listdir(self.experiment_dir) if "csv_records" in s]
        csv_dirs.sort()
        csv_dir = csv_dirs[-1]
        individual_dir = self.experiment_dir + '/generation_{}'.format(generation) +\
            '/individual_{}'.format(trial)
        shutil.move(self.experiment_dir + '/' + csv_dir, individual_dir)
        np.save(individual_dir + '/genetic_string', self.population[trial])

        # save the robot's initial pose in this run
        np.save(individual_dir + '/initial_pose', self.initial_pose)

        wheel_speeds = evolution_utils.get_wheel_speeds(individual_dir)
        np.save(individual_dir + '/wheel_speeds', wheel_speeds)
        _ = evolution_utils.get_robot_path(individual_dir)
        _ = evolution_utils.correct_for_collisions(individual_dir)

        # reload the wheel speeds after correcting for collisions
        correct_wheel_speeds = np.load(individual_dir + '/corrected_wheel_speeds.npy')

        # calculate and save fitness value
        fitness_value = evolution_utils.fitness_function(correct_wheel_speeds)

        print("Fitness: {}".format(fitness_value))
        print("=================")
        np.save(individual_dir + '/fitness_value', fitness_value)

    def run_experiment(self):
        # launch experiment and register status callback
        self.sim = vc.launch_experiment('floreano_0', server='localhost')
        self.sim.register_status_callback(self.on_status)
        self.sim.add_transfer_function(display_episode_tf % "Generation 1, Individual 1")

        for i in range(self.cur_gen, self.generations):
            # Create directory for generation data
            os.mkdir(self.experiment_dir + '/generation_{}'.format(i))

            # Iterate over individuals in a population
            for j in range(60):
                clear_output(wait=True)
                print("Generation {}, Individual {}".format(i + 1, j + 1))
                self.initial_pose = self.set_random_robot_pose()
                self.sim.edit_transfer_function('display_episode_number', display_episode_tf %
                                                "Generation {}, Individual {}".format(i + 1, j + 1))
                genetic_string = ','.join(str(x) for x in self.population[j].ravel())
                self.sim.edit_brain(evolution_utils.brain % genetic_string)
                self.sim.start()

                # run simulation for 40 seconds
                self.wait_condition(10000, lambda x: x['simulationTime'] > self.cur_sim_time + 40,
                                    verbose=True)
                self.sim.pause()

                # reset the robot pose
                self.sim.reset('robot_pose')

                # preserve the data from the previous run
                self.save_simulation_data(i, j)

                self.wait_condition(1000, lambda x: x['state'] == 'paused')
                self.cur_sim_time = self.last_status[0]['simulationTime']

            evolution_utils.get_top_performers(self.experiment_dir + '/generation_{}'.format(i))
            self.population = evolution_utils.evolve_new_generation(self.experiment_dir +
                                                                    '/generation_{}'.format(i))


# random population of 10 binary genetic strings
# it will be only used if there are no previous results stored
population = np.random.randint(2, size=(60, 10, 29))

floreano_experiment = FloreanoExperiment('floreano_0', population, 30)
floreano_experiment.run_experiment()
