import sys
import math
import random
from typing import Tuple, Dict, Union, List
from variables import *
from gym import spaces
# Add Vortex to the system path to be able to use Vortex and vxatp3
vortex_folder = r'C:\CM Labs\Vortex Studio 2020b\bin'
sys.path.append(vortex_folder)
import Vortex
import vxatp3


class Environment:
    def __init__(self):
        """ Initialize all variables """
        self.keyFrameList = None
        self.key_frames_array = None

        self.current_step = 0
        self.reward = 0
        self.rewards_dict = {}

        """File Locations"""
        # Setup File
        self.setup_file = "../VortexFiles/SetupFile/EmptySceneSetup.vxc"
        # Scene Files
        self.scene_file = "../VortexFiles/EmptyScene/NoTruckScene.vxscene"
        # Truck Files
        self.dummy_truck_file = "../VortexFiles/DummyTruck/Mechanism/DummyTruck.vxmechanism"
        self.actual_truck_file = "../Assets/Mechanisms/Semi Truck - Trailer/Semi Truck - Single.vxmechanism"
        self.target_file = "../VortexFiles/TargetGraphic/TargetGraphic.vxmechanism"
        self.recorderFile = "record.vxrec"

        """Setup The Simulation and Load Related Files"""
        # Create the app
        self.application = vxatp3.VxATPConfig.createApplication(self, "Truck Simulation", self.setup_file)
        self.recorder = self.application.getContext().getKinematicRecorder()

        # Choose which camera to view
        if CAMERA == "Scene":
            # Load the scene
            self.vx_scene = self.application.getSimulationFileManager().loadObject(self.scene_file)
            self.scene = Vortex.SceneInterface(self.vx_scene)
            # Load actual Truck
            self.vx_actual_truck = self.application.getSimulationFileManager().loadObject(self.actual_truck_file)
            self.actual_truck = Vortex.MechanismInterface(self.vx_actual_truck)
        elif CAMERA == "Truck":
            # Load actual Truck
            self.vx_actual_truck = self.application.getSimulationFileManager().loadObject(self.actual_truck_file)
            self.actual_truck = Vortex.MechanismInterface(self.vx_actual_truck)
            # Load the scene
            self.vx_scene = self.application.getSimulationFileManager().loadObject(self.scene_file)
            self.scene = Vortex.SceneInterface(self.vx_scene)
        else:
            raise ValueError

        # Load 19 dummy trucks
        self.dummy_trucks = []
        for _ in range(19):
            current_truck = self.application.getSimulationFileManager().loadObject(self.dummy_truck_file)
            self.dummy_trucks.append(Vortex.MechanismInterface(current_truck))

        self.vx_target = self.application.getSimulationFileManager().loadObject(self.target_file)
        self.target = Vortex.MechanismInterface(self.vx_target)

        """Load VHL Files"""
        # VHL to control truck movement
        self.truck_interface = self.actual_truck.findExtensionByName("Vehicle Interface")
        # VHL to get truck observations
        self.observation_interface = self.actual_truck.findExtensionByName("Output Observations")
        self.HUD_interface = self.scene.findExtensionByName("HUD-interface")

        """ Display stuff"""
        # Create a display window
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(WINDOW_X_POS, WINDOW_Y_POS,
                                                                                      WINDOW_X_SIZE, WINDOW_Y_SIZE))

        """ Location of Empty Spot"""
        # Pose of empty spot
        self.empty_spot_var = None
        self.ep = 0

        """ Define Action space and Observation Space"""
        self.action_space = spaces.Box(low=LOWER_BOUND, high=UPPER_BOUND, shape=(NUM_OF_ACTIONS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=LOWER_BOUND, high=UPPER_BOUND, shape=(NUM_OF_STATES,), dtype=np.float32)

        """ Initialize the first scene"""
        self.initialization()

    @staticmethod
    def get_parking_pose() -> List[Vortex.Matrix44]:
        """
        This function gets the positions of the parking spots from the csv file.
        @return: A list of matrices representing the positions of the parking spots.
        """
        # Import csv with spot poses [x, y, z, degree_rx, degree_ry, degree_rz]
        csv_output = np.genfromtxt('example.csv', delimiter=' ')

        all_poses = []
        for line in csv_output:
            x, y, z, drx, dry, drz = line
            rx = np.deg2rad(drx)
            ry = np.deg2rad(dry)
            rz = np.deg2rad(drz)
            # Construct transformation matrix [Rotation Matrix, translation vector; zero vector, 1]
            current_pose_m = Vortex.rotateTo(Vortex.createTranslation(x, y, z), Vortex.VxVector3(rx, ry, rz))
            all_poses.append(current_pose_m)
        return all_poses

    @staticmethod
    def waitForNbKeyFrames(expectedNbKeyFrames, application, keyFrameList) -> None:
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            nbIter += 1

    @staticmethod
    def angles_from_truck_dim(truck_dim: list) -> list:
        """ Find angle to the corners of the truck """
        [pos_x, pos_y, neg_x, neg_y] = truck_dim
        theta = math.atan2(pos_y, pos_x)
        alpha = math.atan2(pos_y, neg_x)
        beta = math.atan2(neg_y, neg_x)
        gamma = math.atan2(neg_y, pos_x)
        angles = [theta, alpha, beta, gamma]
        return angles

    @staticmethod
    def corner_distance_from_truck_dim(truck_dim: list) -> list:
        """ Find distance to the corners of the truck """
        [pos_x, pos_y, neg_x, neg_y] = truck_dim
        one = math.sqrt(pos_x ** 2 + pos_y ** 2)
        two = math.sqrt(neg_x ** 2 + pos_y ** 2)
        three = math.sqrt(neg_x ** 2 + neg_y ** 2)
        four = math.sqrt(pos_x ** 2 + neg_y ** 2)
        lengths = [one, two, three, four]
        return lengths

    @staticmethod
    def find_new_x(old_x: float, length: float, global_angle: float, local_angle: float) -> float:
        return old_x + length * math.cos(global_angle + local_angle)

    @staticmethod
    def find_new_y(old_y: float, length: float, global_angle: float, local_angle: float) -> float:
        return old_y + length * math.sin(global_angle + local_angle)

    @staticmethod
    def check_in_middle(value, small, big) -> bool:
        if small < value < big:
            return True
        else:
            return False

    @staticmethod
    def normalize_states(states: np.array) -> np.array:
        """
        This function normalizes an array of states.
        """
        norm_states = []

        # norms
        dist_norm = 176
        ray_cast_norm = 40
        angle_norm = np.pi
        velocity_norm = 25

        # index value
        all_dist = [0]
        all_raycast = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        all_angles = [1, 2, 3, 4]
        all_velocities = [5, 6, 7, 8, 9, 10]

        for index, val in enumerate(states):
            if index in all_dist:
                norm_states.append(val / dist_norm)
            elif index in all_raycast:
                norm_states.append(val / ray_cast_norm)
            elif index in all_angles:
                norm_states.append(val / angle_norm)
            elif index in all_velocities:
                norm_states.append(val / velocity_norm)
        return norm_states

    @staticmethod
    def make_between_neg_pi_and_pi(angle: float) -> float:
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    @staticmethod
    def deadband(array, deadband: float) -> float:
        """
        Functions that takes an action and a deadband, and stabilizes the action.
        @param array: Action value that we would like to stabilize using the deadband.
        @param deadband: Value of the deadband. Between -deadband and deadband, the values are 0.
        @return: Stabilized action value.
        """
        array = array.numpy()
        if array >= deadband:
            return ((array - deadband)/(1-deadband))**3
        elif array <= -deadband:
            return ((array + deadband)/(1-deadband))**3
        else:
            return 0

    @staticmethod
    def find_distance_and_angle_to_spot(truck: Vortex.Matrix44,
                                        spot: Vortex.Matrix44
                                        ) -> Tuple[float, float]:
        """
        Find distance and angle to spot.
        @param truck: Position of the truck 4x4 matrix
        @param spot: Position of the empty parking spot 4x4 matrix
        @return: The distance and the angle theta between the truck and the empty parking spot.
        """
        xtru, ytru, _ = Vortex.getTranslation(truck)
        xspot, yspot, _ = Vortex.getTranslation(spot)

        p1 = np.array([xtru, ytru])
        p2 = np.array([xspot, yspot])
        distance_to_spot = np.sqrt(np.sum((p1 - p2)**2, axis=0))

        """ Find theta to spot"""
        theta = math.atan2(ytru - yspot, xtru - xspot)
        return distance_to_spot, theta

    @staticmethod
    def transform_velocities(global_linear_vel: Vortex.Vector3, pose: Vortex.Matrix44) -> np.array:
        """
        This function changes global velocities into local velocities using the rotational matrix from the pose
        @param global_linear_vel: Global velocity [x, y, z] with Vortex type
        @param pose: Pose 4x4 matrix [R d; 0 1]
        @return: Local velocity [x, y, z] as a numpy array
        """
        numpy_pose = np.array(pose)
        rotational_matrix = numpy_pose[0:3, 0:3].T
        numpy_g_vel = np.array([global_linear_vel.x, global_linear_vel.y, global_linear_vel.z])
        numpy_l_vel = np.dot(rotational_matrix, numpy_g_vel)
        local_linear_velocity = numpy_l_vel
        return local_linear_velocity

    def check_within_bounds(self, truck_initial_position, start_bounds, truck_dim) -> bool:
        """
        Function to check if truck is within start area
        @param truck_initial_position: list of [x, y, theta] representing truck pose
        @param start_bounds: list of [top, bottom, left, right] representing start area [max_y, min_y, min_x, max_x]
        @param truck_dim: list of [front, left, back, right] representing truck dim from truck origin
        @return True or False on whether truck is inside start area
        """

        # Decompose truck position and start area
        x, y, angle = truck_initial_position
        top, bottom, left, right = start_bounds

        # Find truck corners
        truck_corners = self.find_truck_extremities(x, y, angle, truck_dim)

        # If all four corners are inside start area, then truck is inside start area
        for corner in truck_corners:
            x_within = self.check_in_middle(corner[0], left, right)
            y_within = self.check_in_middle(corner[1], bottom, top)
            if not (x_within and y_within):
                return False
        return True

    def find_truck_extremities(self, x: float, y: float, angle: float, truck_dim: list) -> list:
        """
        Finds the corners of the truck given location of truck
        @param x: Truck's x location
        @param y: Truck's y location
        @param angle: Truck's angle from x axis
        @param truck_dim: [front, left, back, right]
        @return: Corners [bottom left, bottom right, top right, top left]
        """
        # Find corner angles and lengths of truck
        angles = self.angles_from_truck_dim(truck_dim)
        lengths = self.corner_distance_from_truck_dim(truck_dim)

        # Find new corners based on translation and rotation applied
        all_new_points = []
        for i in range(4):
            new_x = self.find_new_x(x, lengths[i], angle, angles[i])
            new_y = self.find_new_y(y, lengths[i], angle, angles[i])
            new_point = (new_x, new_y)
            all_new_points.append(new_point)
        all_new_points = sorted(all_new_points, key=lambda tup: (tup[1], tup[0]))
        return all_new_points

    def random_start_loc(self) -> None:
        """
        Randomly choose Truck Starting Position
        """
        # temporary variables
        min_radius = 15.8
        spot_angle = Vortex.getRotation(self.empty_spot_var).z
        y_mod = 1 if spot_angle == np.pi/2 else -1

        empty_spot_target = self.empty_spot_var
        empty_translation = Vortex.getTranslation(empty_spot_target)
        empty_translation.y += y_mod * 8.5
        empty_spot_target = Vortex.translateTo(empty_spot_target, empty_translation)

        truck_pose_candidate = phi = None

        correct_pos = False
        while not correct_pos:
            radius = min_radius + abs(random.normalvariate(0, 6)) * (0.2 + self.ep/MAX_DIFFICULTY_EPISODE)
            angle = np.pi/2 * y_mod + random.normalvariate(0, 0.4) * (0.2 + self.ep/MAX_DIFFICULTY_EPISODE)
            candidate_x = radius * math.cos(angle)
            candidate_y = radius * math.sin(angle)

            truck_pose_candidate = Vortex.createTranslation(empty_translation.x + candidate_x,
                                                            empty_translation.y + candidate_y, 0)

            dist, phi = self.find_distance_and_angle_to_spot(truck_pose_candidate, empty_spot_target)
            if -20 < empty_translation.x + candidate_x < 65:
                correct_pos = True
        angle_noise = random.normalvariate(0, .1) * min(0.4 + self.ep/MAX_DIFFICULTY_EPISODE, 1)
        self.actual_truck.inputLocalTransform.value = Vortex.rotateTo(truck_pose_candidate,
                                                                      Vortex.VxVector3(0, 0, phi+angle_noise))

    def dummy_truck_placement(self) -> None:
        """
        This function chooses the location of the parking spot and places the dummy trucks in the right places.
        It places the trucks with decreasing rarity over time, so that the difficulty of the problem increases.
        Also, it only places trucks on the same side as the parking spot, as to increase the available space
        that the truck possesses to park.
        """
        # Find all parking poses for the dummy trucks - list of transformation matrices
        all_parking_poses = self.get_parking_pose()

        """Randomly choose empty parking spot"""
        # Choose empty spot location
        empty_spot = random.randint(0, len(all_parking_poses) - 1)
        empty_spot_location = all_parking_poses[empty_spot]
        self.target.inputLocalTransform.value = empty_spot_location
        self.empty_spot_var = empty_spot_location
        del all_parking_poses[empty_spot]

        # Put dummy trucks with decreasing rarity over time
        if empty_spot <= 9:
            for index, truck in enumerate(self.dummy_trucks):
                if random.random() < 0.8 * ((self.ep+1)/MAX_DIFFICULTY_EPISODE) and index <= 8:
                    position = all_parking_poses[index]
                    truck.inputLocalTransform.value = position
                else:
                    truck.inputLocalTransform.value = Vortex.createTranslation(20, 20, -50)

        if empty_spot > 9:
            for index, truck in enumerate(self.dummy_trucks):
                if random.random() < 0.8 * ((self.ep+1)/MAX_DIFFICULTY_EPISODE) and index > 9:
                    position = all_parking_poses[index]
                    truck.inputLocalTransform.value = position
                else:
                    truck.inputLocalTransform.value = Vortex.createTranslation(20, 20, -50)

    def edit_scene(self) -> None:
        """
        This functions places the trucks.
        """
        # Go into editing mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)
        self.dummy_truck_placement()
        self.random_start_loc()
        # Switch to simulation
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

    def initialization(self) -> None:
        """
        This function initializes the scene every episode.
        """
        self.edit_scene()

        """Save this initial keyframe"""
        self.application.update()
        self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList", False)
        self.application.update()
        self.keyFrameList.saveKeyFrame()
        self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
        self.key_frames_array = self.keyFrameList.getKeyFrames()

    def reset(self) -> np.array:
        """
        This function resets the scene and gets the first observations.
        @return: The normalized states.
        """
        # Initialize Reward and Step Count
        self.current_step = 0
        self.reward = 0
        self.rewards_dict = {}

        # Load first key frame
        self.keyFrameList.restore(self.key_frames_array[0])
        self.application.update()

        self.edit_scene()

        state, norm_state = self.get_observations()
        return norm_state

    def ray_cast_distance(self, ray_cast_name: str) -> float:
        """
        This function takes as input the name of a ray cast sensor of the truck, and returns its distance.
        """
        raycast_object = self.observation_interface.getOutputContainer()[ray_cast_name].value
        intersection_point = raycast_object.getOutput('Intersection Point').value
        ray_origin = raycast_object.getOutput('Ray')['Origin'].value
        intersected = raycast_object.getOutput('Has Intersected').value
        if intersected:
            p1 = np.array([intersection_point.x, intersection_point.y, intersection_point.z])
            p2 = np.array([ray_origin.x, ray_origin.y, ray_origin.z])
            dist = np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
        else:
            dist = raycast_object.getInput('Max Distance').value  # should be 100
        return dist

    def render(self, active: bool = True, sync: bool = False) -> None:
        """
        This function controls the rendering of Vortex.
        @param active: Bool for whether to activate the display.
        @param sync: Bool for whether to synchronize to the frame rate. If False, it runs faster.
        """
        # Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        # If active, add a display and activate Vsync
        if active:
            if len(current_displays) == 0:
                self.application.add(self.display)

            if sync:
                self.application.setSyncMode(Vortex.kSyncSoftwareAndVSync)
            else:
                self.application.setSyncMode(Vortex.kSyncNone)

        # If not, remove the current display and deactivate Vsync
        else:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(Vortex.kSyncNone)

    def lowpass_filter(self, input_signal, output_signal, time_constant) -> float:
        """Apply a low-pass filter to a signal."""
        delta_time = self.application.getSimulationTimeStep()
        value = (((delta_time * input_signal) + (time_constant * output_signal))
                 / (delta_time + time_constant))
        return value

    def move(self, action) -> None:
        """
        Move the actual truck based on action given
        @param action: Array = [TBG, Steering]
        @return: Doesn't Return anything
        """
        # Values for lowpass filter
        time_constant = 0.1
        old_truck = self.truck_interface.getInputContainer()["Throttle"].value
        old_gear = self.truck_interface.getInputContainer()["Gear"].value
        old_steer = self.truck_interface.getInputContainer()["Steering"].value

        # Scale the -1 to 1 output to the correct range
        gear = 1 if action[0] >= 0 else -1

        # Apply signal filter to limit sudden movements?
        t_dead = 0.1
        throttle_val = abs(self.deadband(action[0], t_dead))

        # Lowpass for steering
        steer_val = self.lowpass_filter(self.deadband(action[1], 0.25), old_steer, time_constant)

        # Brake
        braking = np.exp(-(2 * action[0].numpy() / t_dead) ** 2)

        # Apply actions
        self.truck_interface.getInputContainer()["Engine Running"].value = True
        self.truck_interface.getInputContainer()["Throttle"].value = throttle_val
        self.truck_interface.getInputContainer()["Brake"].value = braking
        self.truck_interface.getInputContainer()["Steering"].value = steer_val
        self.truck_interface.getInputContainer()["Gear"].value = gear

        self.HUD_interface.getInputContainer()["Throttle"].value = str(round(throttle_val, 4))
        self.HUD_interface.getInputContainer()["Brake"].value = str(round(braking, 4))
        self.HUD_interface.getInputContainer()["Steering"].value = str(round(steer_val, 4))
        self.HUD_interface.getInputContainer()["Gear"].value = str(round(gear, 4))

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, float]]:
        """
        @param action: the action that the truck needs to make.
        @return: state: a numpy array (24 states currently provided by get_observation)
                 reward: a float
                 done: a bool that informs whether the epoch is over (collision, time is up, jack knife, etc)
        """
        self.HUD_interface.getInputContainer()["Step"].value = str(self.current_step)
        self.HUD_interface.getInputContainer()["Frame Rate"].value = "60"

        # Take the action
        self.move(action)

        # Reset the reward every step
        self.reward = 0

        # Update the simulation 4 times
        for _ in range(4):
            self.application.update()

        # Get the states
        state, norm_states = self.get_observations()

        # Make done True when the time of the episode is up
        self.done = False
        if self.current_step > TOTAL_STEPS_PER_EPISODE:
            self.done = True

        # Calculated the reward
        reward, rewards_dict = self.calculate_reward(state)

        self.current_step += 1
        info = rewards_dict
        return norm_states, reward, self.done, info

    def find_phis(self, truck, trailer, spot) -> Tuple[float, float]:
        """
        Finds (1) the angle truck_phi that is the difference between the angle of the truck
        and the angle of the parking spot, and (2) the the angle trailer_phi that is the
        difference between the angle of the trailer and the angle of the parking spot
        @param truck: information about the position and direction of the truck
        @param trailer: information about the position and direction of the trailer
        @param spot: information about the position and direction of the parking spot
        @return: angle phi of the truck and of the trailer
        """
        # Get the angle of the truck, the trailer, and the spot
        _, _, rztru = Vortex.getRotation(truck)
        _, _, rztra = Vortex.getRotation(trailer)
        _, _, rzspot = Vortex.getRotation(spot)

        # Find truck phi
        truck_phi = rztru - rzspot
        truck_phi = self.make_between_neg_pi_and_pi(truck_phi)

        # Find trailer phi
        trailer_phi = rztra - rzspot
        trailer_phi = self.make_between_neg_pi_and_pi(trailer_phi)

        return truck_phi, trailer_phi

    def get_observations(self) -> Tuple[np.array, np.array]:
        """
        This function retrieves the state observation from Vortex. The states are all relative to their destination,
        so that it can generalize to any position.
        @return: the state of the Vortex simulation at that instant, and the normalized state for easier learning.
        """
        # Find dist_to_spot and theta using truck's pose and spot's pose
        truck_world_pose = self.observation_interface.getOutputContainer()["Truck Pose"].value
        dist_to_spot, theta = self.find_distance_and_angle_to_spot(truck_world_pose, self.empty_spot_var)

        # Find phi for truck and phi for trailer using truck's pose, trailer's pose and spot's pose
        trailer_world_pose = self.observation_interface.getOutputContainer()["Trailer Pose"].value
        phi_for_truck, phi_for_trailer = self.find_phis(truck_world_pose, trailer_world_pose, self.empty_spot_var)

        # Get hinge angle
        hinge_angle = self.observation_interface.getOutputContainer()["Hinge Angle"].value
        hinge_angle = self.make_between_neg_pi_and_pi(hinge_angle)

        # Get truck's x, y, rz velocities
        lv_truck = self.observation_interface.getOutputContainer()["Truck Linear Velocity"].value
        vxtru, vytru, _ = self.transform_velocities(lv_truck, truck_world_pose)
        _, _, wztru = self.observation_interface.getOutputContainer()["Truck Angular Velocity"].value

        # Get trailer's x, y, rz velocities
        lv_trailer = self.observation_interface.getOutputContainer()["Trailer Linear Velocity"].value
        vxtra, vytra, _ = self.transform_velocities(lv_trailer, trailer_world_pose)
        _, _, wztra = self.observation_interface.getOutputContainer()["Trailer Angular Velocity"].value

        # Get ray cast distance
        truck_front_rc = self.ray_cast_distance("Raycast Truck Front")
        truck_front_right_rc = self.ray_cast_distance("Raycast Truck Front Right")
        truck_front_left_rc = self.ray_cast_distance("Raycast Truck Front Left")
        truck_right_rc = self.ray_cast_distance("Raycast Truck Right")
        truck_left_rc = self.ray_cast_distance("Raycast Truck Left")
        trailer_left_rc = self.ray_cast_distance("Raycast Trailer Left")
        trailer_right_rc = self.ray_cast_distance("Raycast Trailer Right")
        trailer_back_rc = self.ray_cast_distance("Raycast Trailer Back")
        trailer_back_left_rc = self.ray_cast_distance("Raycast Trailer Back Left")
        trailer_back_right_rc = self.ray_cast_distance("Raycast Trailer Back Right")

        all_states = [dist_to_spot, theta, phi_for_truck, phi_for_trailer, hinge_angle,
                      vxtru, vytru, wztru, vxtra, vytra, wztra,
                      truck_front_rc, truck_front_right_rc, truck_front_left_rc, truck_right_rc, truck_left_rc,
                      trailer_back_rc, trailer_back_right_rc, trailer_back_left_rc, trailer_right_rc, trailer_left_rc]

        all_states_norm = self.normalize_states(all_states)

        all_states = np.array(all_states)
        all_states_norm = np.array(all_states_norm)

        return all_states, all_states_norm

    def add_reward(self, reward_amount: float, reward_type: str = 'UNNAMED') -> None:
        """
        Adds key reward_name with value reward_amount to the rewards_dict
        @param reward_amount: the number of reward for its reward type.
        @param reward_type: the name of the reward that is added to.
        """

        self.reward += reward_amount

        # If the reward_type key doesn't yet exist in the dictionary, create it.
        if reward_type not in self.rewards_dict:
            self.rewards_dict[reward_type] = 0
        self.rewards_dict[reward_type] += reward_amount

    def calculate_reward(self, state: np.array) -> Tuple[int, Dict[str, float]]:
        """
        This function calculates the reward that the model receives at every step, and adds it to a dictionary
        detailing which "type" of reward had which value. For example, after a few steps you could have a dict:
        reward_dict = {"Reward for being close": -182.3, "Reward for speed": -19.3, "Collision Punishment": -33}

        The reward is calculated by:
        - Giving higher negative reward the farther the truck is from the parking spot
        - Giving higher negative reward the slower the truck goes
        - Giving higher negative reward the if the truck is far and has a high phi angle.
        - Giving negative reward every step where the truck is currently colliding
        - Giving negative reward every step where the truck is currently in jackknife

        @param state: states used to calculate the reward.
        @return: reward and reward dictionary
        """
        # States
        dist_to_spot, theta, phi_for_truck, phi_for_trailer, hinge_angle, \
        vxtru, vytru, wztru, vxtra, vytra, wztra = state[:11]

        # Check if collided
        collision = True if self.observation_interface.getOutputContainer()['Truck Collided'].value else False
        # Check if jackknifed
        jack_knife = True if abs(hinge_angle) > JACKKNIFE_ANGLE else False

        # Punish for distance to spot
        self.add_reward(-CLOSE_DISTANCE_REWARD * dist_to_spot, "Reward for being close")

        # Reward for speed
        max_rew = 25*SPEED_REWARD
        self.add_reward(-max_rew + SPEED_REWARD * (vxtru + vytru + vxtra + vytra), "Reward for speed")

        # Angle-distance reward
        max_rew = LOW_ANGLE_REWARD
        dist_factor = 7
        phi_factor = 1.5
        self.add_reward(-max_rew + LOW_ANGLE_REWARD * np.exp(-((dist_to_spot/dist_factor) ** 2)
                                                             - (phi_factor*phi_for_trailer ** 2)),
                        "Reward for low angle of trailer")

        # Punishment for collisions
        if collision:
            self.add_reward(COLLISION_PUNISH, "Collision Punishment")
        if jack_knife:
            self.add_reward(JACK_KNIFE_PUNISHMENT, "Jack knife Punishment")
        return self.reward, self.rewards_dict
