import gym
import time
import numpy as np
import copy
import random
import heapq
""" Setup some logging. """

import logging
import colorlog

logger = logging.getLogger("RoboticWareHouse")

handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    fmt=('%(log_color)s[%(asctime)s %(levelname)8s] --'
         ' %(message)s (%(filename)s:%(lineno)s)'),
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logger.addHandler(handler)

# logger.setLevel()

dynamic_import = {"cv2": None}


class Package(object):
    def __init__(self, identifier: int, start: [], dropoff: [], spawn: int,
                 map_reference: "RoboticWarehouse"):
        self.identifier = identifier
        self.start = start
        self.dropoff = dropoff
        self.spawn = spawn
        self.map_reference = map_reference


class Robot(object):
    def __init__(self, position: [], packages: []):
        self.position = position
        self.packages = packages

        self.reservations = set()


class RoboticWarehouse(gym.Env):
    TILE_ID = 0
    TILE = [0, 0]
    SHELF_ID = 1
    SHELF = [1, 0]
    PACKAGE_ID = 2
    DROP_ID = 4
    DROP = [4, 0]

    DOWN_INSTRUCTION = 0
    LEFT_INSTRUCTION = 1
    UP_INSTRUCTION = 2
    RIGHT_INSTRUCTION = 3
    PICKUP_INSTRUCTION = 4
    DROP_INSTRUCTION = 5

    UP = [1, 0]
    DOWN = [-1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]

    def __init__(
            self,
            robots: int = 1,  # Number of robots
            capacity: int = 1,  # Number of packages robot can carry
            spawn: int = 10,  # Initial packages spawned
            shelve_length: int = 2,  # length of a shelf
            shelve_height: int = 2,  # number of shelves in a column (bad name?)
            shelve_width: int = 2,  # number of shelves in a row (bad name?)
            shelve_throughput: int = 1,  # number of robots that can pass
            cross_throughput: int = 1,  # number of robots that can pass
            seed: int = 103,  # Seed used to choose package spawns
            periodicity_lower: int = 400,
            periodicity_upper: int = 1000
    ):  # How many places to spawn packages
        """ Number of packages a robot can hold. """
        self.capacity = capacity
        """ Remember this for environment resets. """
        self.initial_spawn = spawn
        """ Keep track of how many steps have been taken. """
        self.steps = 0
        """ Keep track of num_robots. """
        self.num_robots = robots
        """ Seed to make sure same behavior. """
        self.seed = seed
        """ 
        The map. 
        ---------------------------
        I           C             I
        I    SS  SS  SS  SS  SS   I
        I C  SS  SS  SS  SS  SS C I
        I    SS  SS  SS  SS  SS   I
        I           C             I
        I    SS  SS  SS  SS  SS   I
        I C  SS  SS  SS  SS  SS C I
        I    SS  SS  SS  SS  SS   I
        I           C             I
        --------------------------I

        S constellations = shelves_width * shelve_height
        S = S constellations * shelve_length
        Distance between S constellations on a row = shelve_throughput
        Other distance = cross_throughput

        All aisles marked with a C is a cross aisle (Do you agree?)
        """

        self.map_width = 2 * shelve_width + 2 * cross_throughput + (
            shelve_width - 1) * shelve_throughput
        self.map_height = shelve_length * shelve_height + (
            shelve_height + 1) * cross_throughput

        #TODO: Double check my logic on this one (It appears to be correct though)
        self.shelve_positions = [
            (y, x) for y in range(self.map_height)
            # This checks if were at a y position where a shelve can occur
            if (y - cross_throughput) % (shelve_length + cross_throughput) <
            shelve_length and y >= cross_throughput
            for x in range(self.map_width)
            # This checks if were at a x position where a shelve can occur
            if (x - cross_throughput) % (2 + shelve_throughput) < 2
            and self.map_width - cross_throughput > x and x >= cross_throughput
        ]
        """ This makes setup faster. """
        self.shelve_positions = set(self.shelve_positions)

        self.floor_positions = [(y, x) for y in range(self.map_height)
                                for x in range(self.map_width)
                                if (y, x) not in self.shelve_positions]
        """ To make random choices O(1). """
        self.shelve_positions = list(self.shelve_positions)
        """ Drop positions. (These must be initialized before setup env.) """
        # TODO: Make these depend on map
        self.drop_positions = [[0, 0], [0, 1], [0, 2]]
        """ To make sure same thing happends. """
        random.seed(seed)
        if spawn > len(self.shelve_positions):
            raise Exception(
                "Not enough shelves {} to spawn {} packages".format(
                    len(self.shelve_positions), spawn))
        """ Package spawn positions. """
        self.package_spawn_positions = list(
            random.sample(self.shelve_positions, spawn))
        """ Make sure there is a periodicity pattern to the positions aswell (Something something can learn? :)) """
        self.package_spawn_times = [[
            random.randint(periodicity_lower, periodicity_upper),
            random.randint(periodicity_lower, periodicity_upper), i
        ] for i in range(len(self.package_spawn_positions))]
        """ Make into a heap. """
        heapq.heapify(self.package_spawn_times)
        """ To make sure not same thing happends from here (dont want all simulations to be equal ^^. """
        random.seed(time.time())

        self.__setup_env()

        self.__actions = [
            self.__move_down, self.__move_left, self.__move_up,
            self.__move_right, self.__pickup_package, self.__drop_package
        ]

        self.action_space = ActionSpace(robots, len(self.__actions))
        # TODO: ...
        self.observation_space = None
        self.round_collisions = 0
        self.round_dropoffs = []

    def __setup_env(self) -> None:
        """ Setup map first. There is no need to have a sparse map. (Since they will be relativley small)"""
        self.shelve_positions = set(self.shelve_positions)
        self.map = []
        for y in range(self.map_height):
            row = []
            for x in range(self.map_width):
                if (y, x) in self.shelve_positions:
                    row.append([RoboticWarehouse.SHELF_ID, 0])
                else:
                    row.append([RoboticWarehouse.TILE_ID, 0])
            self.map.append(row)
        self.shelve_positions = list(self.shelve_positions)
        """ Add drops to map for visually pleasing graphics ^^. """
        for y, x in self.drop_positions:
            self.map[y][x][0] = RoboticWarehouse.DROP_ID
        """ 
        Keep track of packages and spawn initial packages. 
        
        Important that package handling is performant since its probably going to be
        the bottle neck, or robot movements?

        Need to be able to Add packages
        Need to be able to Remove packages
        Need to be able to get a representation of 
        all free packages to send to user

        Num possible package positions is num shelve positions

        Constraints:
            Cannot add a package where another one is (No stacking)


        Idea:
            Use a dict (hashmap)
                Key:  Some identifier (E.g) random int
                Value: Package Object

            Complexities:
                P = packages / shelves
                Adding O(k) k = (1 / (1 - P)) (Assuming binomial distribution)
                Removing O(1)
                Representatable O(p) where p = number of packages

            This is ok if packages are sparse

            Why dict and not set?
            Because a identifier will be added to the grid
            and used to reference the package
        
        """

        self.packages = {}
        # for _ in range(self.initial_spawn):
        # if len(self.packages) == len(self.package_spawn_positions):
        # logger.error(
        # "Cannot spawn more packages -- No Free positions -- Number Packes: {} -- Number Shelves: {}".
        # format(
        # len(self.packages), len(self.package_spawn_positions)))
        # break

        # identifier = np.random.randint(0, 2**32)
        # while identifier in self.packages:
        # identifier = random.randint(0, 2**32)

        # y, x = random.choice(self.package_spawn_positions)
        # while self.map[y][x][0] != RoboticWarehouse.SHELF_ID:
        # y, x = random.choice(self.package_spawn_positions)

        # self.packages[identifier] = Package(identifier, [y, x],
        # random.choice(
        # self.drop_positions),
        # self.steps, self)
        # self.map[y][x][0], self.map[y][x][
        # 1] = RoboticWarehouse.PACKAGE_ID, identifier
        """Placing Robots"""
        self.robots = []
        for robot in range(self.num_robots):
            y, x = random.choice(self.floor_positions)
            self.robots.append(Robot([y, x], []))
            """ One more robot standing at that position. """
            self.map[y][x][1] += 1
        """ For Graphics. """
        self.bitmap = np.zeros((self.map_height, self.map_width, 3))
        self.colors = {
            RoboticWarehouse.TILE_ID: np.array([.0, .0, .0]),
            RoboticWarehouse.SHELF_ID: np.array([0.5, 0.2, 0.05]),
            RoboticWarehouse.PACKAGE_ID: np.array([0.0, 0.8, 0]),
            RoboticWarehouse.DROP_ID: np.array([1.0, 0, 1.0]),
        }

    def reset(self) -> ('robots', 'packages'):
        self.steps = 0
        self.__setup_env()
        return (self.robots, list(self.packages.values()))

    def branch(self) -> "RoboticWarehouse":
        """ Naive implementation for algorithms that need to search future states """
        return copy.deepcopy(self)

    def close(self) -> None:
        """ Do all eventual cleanup here. """
        pass

    def step(self, actions: np.ndarray
             ) -> (('robots', 'packages'), np.float64, bool, None):
        """ 
            Action: [Robotic Action...]
                Robotic Action: X in [0, 5]
                    X == 0 = down
                    X == 1 = left
                    X == 2 = up
                    X == 3 = right
                    X == 4 = pickup package
                    X == 5 = drop package

            First spawn new packages
        """
        self.round_collisions = 0
        self.round_dropoffs = []
        """ Decrement all spawn-times. """
        for i in range(len(self.package_spawn_times)):
            """ Subtracting all with 1 will preserved heap structure. """
            self.package_spawn_times[i][0] = self.package_spawn_times[i][0] - 1

        while self.package_spawn_times[0][0] <= 0:
            package = heapq.heappop(self.package_spawn_times)

            identifier = np.random.randint(0, 2**32)
            while identifier in self.packages:
                identifier = random.randint(0, 2**32)

            y, x = self.package_spawn_positions[package[2]]
            if self.map[y][x] == RoboticWarehouse.SHELF:
                self.packages[identifier] = Package(identifier, [y, x],
                                                    random.choice(
                                                        self.drop_positions),
                                                    self.steps, self)
                self.map[y][x][0], self.map[y][x][
                    1] = RoboticWarehouse.PACKAGE_ID, identifier
            """ Reset Spawn Timer. """
            package[0] = package[1]
            """ Add to queue. """
            heapq.heappush(self.package_spawn_times, package)
        """ 
        Now perform all actions and update map. 

        Important! 
            How to handle collisions
                Example Edge Case

                    R -> E <- R

                What robot gets to go to E?

                I'll execute all actions in order so the robot
                that is first in order gets to go to E and the other robot
                will simply not move.

            If a robot issues drop or pickup in a position where it is not 
            supposed to be able to do that, nothing happends.

        """
        reward = 0
        for r, action in enumerate(actions):
            reward += self.__actions[action](self.robots[r])
        """ Increment steps. """
        self.steps += 1
        """ Maybe there is some better choice for storing packages.. """
        return (self.robots, list(self.packages.values())), reward, False, None

    def __move_up(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.UP)

    def __move_down(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.DOWN)

    def __move_right(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.RIGHT)

    def __move_left(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.LEFT)

    def __pickup_package(self, robot: Robot) -> int:
        """ Don't pick up anything if capacity is full. """
        if len(robot.packages) >= self.capacity:
            return 0
        """ Currently only picks in a grid.. maybe add diagonals?. """

        possibilities = []
        for adjacent_position in [[
                RoboticWarehouse.UP[0] + robot.position[0],
                RoboticWarehouse.UP[1] + robot.position[1]
        ], [
                RoboticWarehouse.DOWN[0] + robot.position[0],
                RoboticWarehouse.DOWN[1] + robot.position[1]
        ], [
                RoboticWarehouse.LEFT[0] + robot.position[0],
                RoboticWarehouse.LEFT[1] + robot.position[1]
        ], [
                RoboticWarehouse.RIGHT[0] + robot.position[0],
                RoboticWarehouse.RIGHT[1] + robot.position[1]
        ]]:
            y, x = adjacent_position
            if self.in_map(
                    y, x) and self.map[y][x][0] == RoboticWarehouse.PACKAGE_ID:
                """ 
                    Now add package to robot. 

                    Make sure
                        1: Package is also removed from map
                        2: Package is also removed from the free map
                """
                """ Add package to robot. (Only add To positon, will never need from.. (I hope)) """
                possibilities.append((y, x))

        for y, x in possibilities:
            package = self.packages[self.map[y][x][1]]
            if package in robot.reservations and len(
                    robot.packages) <= self.capacity:
                robot.packages.append(package)
                """ Remove package from free packages. """
                del self.packages[self.map[y][x][1]]
                """ Remove package from map. """
                self.map[y][x][0], self.map[y][x][
                    1] = RoboticWarehouse.SHELF_ID, 0
                """ Remove from reservations. """
                robot.reservations.remove(package)

        return 0

    def __drop_package(self, robot: Robot) -> int:
        """ Don't try to drop anything if there is nothing. """
        if len(robot.packages) == 0:
            return 0

        score = 0
        for adjacent_position in [[
                RoboticWarehouse.UP[0] + robot.position[0],
                RoboticWarehouse.UP[1] + robot.position[1]
        ], [
                RoboticWarehouse.DOWN[0] + robot.position[0],
                RoboticWarehouse.DOWN[1] + robot.position[1]
        ], [
                RoboticWarehouse.LEFT[0] + robot.position[0],
                RoboticWarehouse.LEFT[1] + robot.position[1]
        ], [
                RoboticWarehouse.RIGHT[0] + robot.position[0],
                RoboticWarehouse.RIGHT[1] + robot.position[1]
        ]]:
            dropped_packages = []
            for package_index in range(len(robot.packages)):
                if robot.packages[package_index].dropoff == adjacent_position:
                    score += 1
                    self.round_dropoffs.append(robot.packages[package_index])
                    dropped_packages.append(package_index)
            """ Remove all packages we have dropped. """
            offset = 0
            for drop in dropped_packages:
                del robot.packages[drop - offset]
                offset += 1

        return score

    def __move_direction(self, robot: Robot, direction: np.ndarray) -> int:
        oy, ox = robot.position
        y, x = robot.position[0] + direction[0], robot.position[1] + direction[1]
        if self.in_map(y, x) and self.map[y][x][0] == RoboticWarehouse.TILE_ID:
            """ One more robot now at that tile. """
            self.map[y][x][1] += 1
            """ One less robot at this tile. """
            self.map[oy][ox][1] = self.map[oy][ox][1] - 1
            """ Update Robot position. """
            robot.position[0], robot.position[1] = y, x
            """ if collision. """
            if self.map[y][x][1] > 1:
                self.round_collisions += 1
        return 0

    def in_map(self, y: int, x: int):
        return (0 <= x < self.map_width and 0 <= y < self.map_height)

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        global dynamic_import
        #TODO: make this less inefficient
        """ 
        Don't want to import this top level since it is a rather 
        big dependency and not everyone cares about rendering
        """
        if dynamic_import["cv2"] == None:
            dynamic_import["cv2"] = __import__("cv2")
        """ Fill Map. """
        for y in range(self.map_height):
            for x in range(self.map_width):
                at_pos = self.map[y][x]
                if self.map[y][x][0] in self.colors:
                    self.bitmap[y][x] = self.colors[self.map[y][x][0]]

        robot_package_color = np.array([0, 0.8, 0.8])
        robot_color = np.array([0, 0, 0.8])
        """ Place robots. """
        for robot in self.robots:
            y, x = robot.position
            if robot.packages:
                self.bitmap[y][x] = robot_package_color
            else:
                self.bitmap[y][x] = robot_color

        ratio = self.map_width / self.map_height
        y_dim, x_dim = min(100 * self.map_height, 800), min(
            100 * self.map_width * ratio, 800 * ratio)

        if x_dim > y_dim and x_dim > 1000:
            y_dim *= (800 / x_dim)
            x_dim *= (800 / x_dim)

        dynamic_import["cv2"].imshow("Game", dynamic_import["cv2"].resize(
            self.bitmap, (int(x_dim), int(y_dim))))
        dynamic_import["cv2"].waitKey(1)

        return self.bitmap

    def __str__(self) -> str:
        return "RoboticWarehouse"


class ActionSpace(gym.spaces.MultiDiscrete):
    def __init__(self, robots: int, categories: int):
        gym.spaces.MultiDiscrete.__init__(self, np.ones(robots) * categories)
