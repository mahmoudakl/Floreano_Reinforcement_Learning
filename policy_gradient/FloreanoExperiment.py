import os
import time
import rospy
import shutil
import logging

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

    def __init__(self, experiment_name, num_episodes):
        self.experiment_name = experiment_name
        self.experiment_dir = os.environ['HOME'] + '/.opt/nrpStorage/' + self.experiment_name
        self.last_status = [None]
        self.sim = None
        self.initial_pose = np.array([])
        self.num_episodes = num_episodes

        self._set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState)

        # keep track of the simulation time, as the reset call is not executed at the precise time
        self.cur_sim_time = 0

        # remove any previously saved csv_recorders' data
        csv_dirs = [s for s in os.listdir(self.experiment_dir) if "csv_records" in s]
        for directory in csv_dirs:
            shutil.rmtree(self.experiment_dir + '/' + directory)

        # Check if evolutionary experiment has been run before
        # If yes, load previously evolved population and continue
        previous_episodes = [int(s.split('_')[1]) for s in os.listdir(self.experiment_dir)
                             if "episode" in s]
        previous_episodes.sort()
        if len(previous_episodes) == 0:
            # start from scratch
            self.cur_episode = 0
        else:
            # TODO: Load previous neural network
            self.cur_episode = previous_episodes[-1]
            last_episode_dir = self.experiment_dir + '/episode_{}'.format(self.cur_episode)

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

    def wait_condition(self, timeout, condition):
        """
        Helper method that blocks for the timeout specified, until the condition
        given has been fulfilled
        """
        start = time.time()
        while time.time() < start + timeout:
            time.sleep(1)
            if self.last_status[0] is not None:
                if condition(self.last_status[0]):
                    return
        raise Exception('Condition check failed')

    def on_status(self, status):
        """Prepends the most recent ROS status message to the last_status array"""
        self.last_status[0] = status

    @staticmethod
    def get_wheel_speeds(episode_dir, corrected=False):
        """
        Loads the saved wheel speeds data for a specific episode. The loaded data can either be the
        raw data recorded from the experiment, or the processed data that corrects for collisions

        :param episode_dir: The directory containing the individual's saved results
        :param corrected: Flag indicating whether the data loaded should be the raw data, or the
                          processed data that corrects for collisions
        """
        if not corrected:
            file_path = episode_dir + '/wheel_speeds.csv'
        else:
            file_path = episode_dir + '/corrected_wheel_speeds.csv'
        wheel_speeds = [i.strip().split(',') for i in open(file_path).readlines()][2:402]
        return wheel_speeds

    @staticmethod
    def get_robot_path(episode_dir):
        """

        :param episode_dir: The directory containing the individual's saved results
        """
        file_path = episode_dir + '/robot_position.csv'
        robot_path = [i.strip().split(',') for i in open(file_path).readlines()][1:401]
        np.save(episode_dir + '/robot_path', robot_path)
        return robot_path

    def correct_for_collisions(self, episode_dir):
        """
        Loops through the robot path and sets wheel speeds to zeros if the robot exceeded certain
        thresholds in the x- and y-positions. These thresholds indicate that the robot has hit the
        wall

        :param episode_dir: The directory containing the individual's saved results
        """

        # convert wheel_speeds to a numpy array to be able to broadcast zeros if a collision is
        # detected
        wheel_speeds = np.asarray(self.get_wheel_speeds(episode_dir))
        robot_path = self.get_robot_path(episode_dir)
        x_axis = [x[0] for x in robot_path]
        y_axis = [y[1] for y in robot_path]
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
        np.save(episode_dir + '/corrected_wheel_speeds', wheel_speeds)
        return collision

    @staticmethod
    def compute_reward(wheel_speeds):
        """
        Calculates the fitness function described in the Floreano paper based on the saved wheel
        speeds
        """
        left_wheel = [float(t[1]) for t in wheel_speeds]
        right_wheel = [float(t[2]) for t in wheel_speeds]
        reward = 0
        for i in range(len(left_wheel)):
            if left_wheel[i] > 0 and right_wheel[i] > 0:
                reward += (left_wheel[i] + right_wheel[i])
        return reward / float(2 * len(left_wheel))

    def save_simulation_data(self, episode):
        """
        Saves the simulation csv data to the respective individual's directory inside the
        experiment's directory
        """
        csv_dirs = [s for s in os.listdir(self.experiment_dir) if "csv_records" in s]
        csv_dirs.sort()
        csv_dir = csv_dirs[-1]
        episode_dir = self.experiment_dir + '/episode_{}'.format(episode)
        shutil.move(self.experiment_dir + '/' + csv_dir, episode_dir)

        # save the robot's initial pose in this run
        np.save(episode_dir + '/initial_pose', self.initial_pose)

        wheel_speeds = self.get_wheel_speeds(episode_dir)
        np.save(episode_dir + '/wheel_speeds', wheel_speeds)
        robot_path = self.get_robot_path(episode_dir)
        collision = self.correct_for_collisions(episode_dir)

        # reload the wheel speeds after correcting for collisions
        correct_wheel_speeds = np.load(episode_dir + '/corrected_wheel_speeds.npy')

        # calculate and save fitness value
        reward = self.compute_reward(correct_wheel_speeds)

        print "Fitness: {}".format(reward)
        np.save(episode_dir + '/fitness_value', reward)

    def run_experiment(self):
        # launch experiment and register status callback
        self.sim = vc.launch_experiment('floreano_1', server='localhost')
        self.sim.register_status_callback(self.on_status)
        self.sim.add_transfer_function(display_episode_tf % "Episode 1")

        for i in range(self.cur_episode, self.num_episodes):
            # Create directory for generation data
            os.mkdir(self.experiment_dir + '/episode_{}'.format(i + 1))

            clear_output(wait=True)
            print "Episode # {}".format(i + 1)
            self.initial_pose = self.set_random_robot_pose()

            # Display episode number in the NRP Frontend
            self.sim.edit_transfer_function('display_episode_number', display_episode_tf %
                                            "Episode # {}".format(i + 1))

            # Start the experiment
            self.sim.start()

            # run simulation for 40 seconds. Pause the simulation and save data after 40s have
            # elapsed
            self.wait_condition(10000, lambda x: x['simulationTime'] > self.cur_sim_time + 40)
            self.sim.pause()
            self.save_simulation_data(i)

            # reset the robot pose
            self.sim.reset('robot_pose')
            self.wait_condition(1000, lambda x: x['state'] == 'paused')
            self.cur_sim_time = self.last_status[0]['simulationTime']


floreano_experiment = FloreanoExperiment('floreano_0', 3000)
floreano_experiment.run_experiment()
