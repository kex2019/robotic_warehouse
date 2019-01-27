import gym
import numpy as np
import copy
import random
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


class RoboticWarehouse(gym.Env):
    FREE = 0
    SHELF = 1
    PACKAGE = lambda x: (2, x)
    ROBOT = lambda x: (3, x)

    UP = np.array([1, 0])
    DOWN = np.array([-1, 0])
    LEFT = np.array([0, -1])
    RIGHT = np.array([0, 1])

    def __init__(
            self,
            robots: int = 1,  # Number of robots
            capacity: int = 1,  # Number of packages robot can carry
            speed: np.float64 = 1,  # Speed of Robot
            spawn: int = 10,  # Initial packages spawned
            spawn_rate: np.float64 = 1,  #  Packages spawned every time t
            shelve_length: int = 2,  # length of a shelf
            shelve_height: int = 2,  # number of shelves in a column (bad name?)
            shelve_width: int = 2,  # number of shelves in a row (bad name?)
            shelve_throughput: int = 1,  # number of robots that can pass
            cross_throughput: int = 1):  # number of robots that can pass
        """ Remember this for environment resets. """
        self.inital_spawn = spawn
        self.spawn = spawn
        """ Remember this for incremental spawn. """
        self.spawn_rate = spawn_rate
        """ Keep track of how many steps have been taken. """
        self.steps = 0
        """ Keep track of num_robots. """
        self.num_robots = robots
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
        """ There is no need to have a sparse map. (Since they will be relativley small)"""
        self.map = []
        for y in range(self.map_height):
            row = []
            for x in range(self.map_width):
                row.append(
                    RoboticWarehouse.FREE if (y, x) not in
                    self.shelve_positions else RoboticWarehouse.SHELF)
            self.map.append(row)
        """ To make random choices O(1). """
        self.shelve_positions = list(self.shelve_positions)

        self.__setup_env()

        self.__actions = [
            self.__move_down, self.__move_left, self.__move_up,
            self.__move_right, self.__pickup_package, self.__drop_package
        ]

        self.action_space = ActionSpace(robots, len(self.__actions))
        # TODO: ...
        self.observation_space = None

    def __setup_env(self) -> None:
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
                Value: Tuple (From, To)

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
        for _ in range(self.inital_spawn):

            if len(self.packages) == len(self.shelve_positions):
                logger.error(
                    "Cannot spawn more packages -- No Free positions -- Number Packes: {} -- Number Shelves: {}".
                    format(len(self.packages), len(self.shelve_positions)))
                break

            identifier = np.random.randint(0, 2**32)
            while identifier in self.packages:
                identifier = random.randint(0, 2**32)

            y, x = random.choice(self.shelve_positions)
            while self.map[y][x] != RoboticWarehouse.SHELF:
                y, x = random.choice(self.shelve_positions)

            # Where to drop of package
            TO = np.array([0, 0])

            self.packages[identifier] = (np.array([y, x]), TO)
            self.map[y][x] = RoboticWarehouse.PACKAGE(identifier)
            self.spawn -= 1
        """Placing Robots"""
        self.robots = []
        for robot in range(self.num_robots):

            if len(self.robots) == len(self.floor_positions):
                logger.error(
                    "Cannot spawn more robots -- No Free positions -- Number robots: {} -- Number floor positions: {}".
                    format(len(self.robots), len(self.floor_positions)))
                break

            y, x = random.choice(self.floor_positions)
            while self.map[y][x] != RoboticWarehouse.FREE:
                y, x = random.choice(self.floor_positions)
            """
                A Robot is a tuple
                    0: 
                        0: Robot Y position
                        1: Robot X position
                    1: 
                        X: [Package]
                            Package: [From, To]
                                From: [Y, X]
                                To: [Y, X]
            """
            self.robots.append((np.array([y, x]), np.array([])))
            self.map[y][x] = RoboticWarehouse.ROBOT(robot)

    def reset(self) -> None:
        self.spawn = self.inital_spawn
        self.__setup_env()

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
        self.spawn += self.spawn_rate
        """ Monte carlo spawning of packages  """
        for _ in range(int(self.spawn)):
            if len(self.packages) == len(self.shelve_positions):
                logger.error(
                    "Cannot spawn more packages -- No Free positions -- Number Packes: {} -- Number Shelves: {}".
                    format(len(self.packages), len(self.shelve_positions)))
                break

            identifier = np.random.randint(0, 2**32)
            while identifier in self.packages:
                identifier = random.randint(0, 2**32)

            y, x = random.choice(self.shelve_positions)
            while self.map[y][x] != RoboticWarehouse.SHELF:
                y, x = random.choice(self.shelve_positions)

            TO = np.array([0, 0])

            self.packages[identifier] = (np.array([y, x]), TO)
            self.map[y][x] = RoboticWarehouse.PACKAGE(identifier)

            self.spawn -= 1
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

        How do we handle different speeds?
        """
        for r, action in enumerate(actions):
            self.__actions[action](self.robots[r])

        return (self.robots, self.packages.values()), 0, False, None

    def __move_up(self, robot: tuple) -> None:
        return self.__move_direction(robot, RoboticWarehouse.UP)

    def __move_down(self, robot: tuple) -> None:
        return self.__move_direction(robot, RoboticWarehouse.DOWN)

    def __move_right(self, robot: tuple) -> None:
        return self.__move_direction(robot, RoboticWarehouse.RIGHT)

    def __move_left(self, robot: tuple) -> None:
        return self.__move_direction(robot, RoboticWarehouse.LEFT)

    def __pickup_package(self, robot: tuple) -> None:
        print("Tried to pick up Package")

    def __drop_package(self, robot: tuple) -> None:
        print("Tried to drop package")

    def __move_direction(self, robot: tuple, direction: np.ndarray) -> None:
        oy, ox = robot[0]
        y, x = robot[0] + direction
        if self.__within_map(y, x) and self.map[y][x] == RoboticWarehouse.FREE:
            self.map[y][x], self.map[oy][ox] = self.map[oy][
                ox], RoboticWarehouse.FREE

            robot[0][0], robot[0][1] = y, x

    def __within_map(self, y: int, x: int):
        return (0 <= x < self.map_width and 0 <= y < self.map_height)

    def seed(self, seed: int) -> None:
        """ To make sure initialization is deterministic: set seeds yourself. """
        random.seed(seed)
        np.random.seed(seed)

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        global dynamic_import
        #TODO: make this less inefficient
        """ 
        Don't want to import this top level since it is a rather 
        big dependency and not everyone cares about rendering
        """
        if dynamic_import["cv2"] == None:
            dynamic_import["cv2"] = __import__("cv2")

        free_robot = 5
        robot_with_package = 6
        colors = {
            RoboticWarehouse.FREE: np.array([.0, .0, .0]),
            RoboticWarehouse.SHELF: np.array([0.5, 0.2, 0.05]),
            2: np.array([0.0, 0.8, 0]),
            free_robot: np.array([0, 0, 0.8]),
            robot_with_package: np.array([0.0, 0.8, 0.8]),
        }

        bitmap = np.zeros((self.map_height, self.map_width, 3))

        for y in range(self.map_height):
            for x in range(self.map_width):
                at_pos = self.map[y][x]

                if type(at_pos) != tuple:
                    bitmap[y][x] = colors[at_pos]
                elif at_pos[0] == 2:
                    bitmap[y][x] = colors[2]
                elif at_pos[0] == 3 and self.robots[at_pos[1]][1]:
                    bitmap[y][x] = colors[robot_with_package]
                elif at_pos[0] == 3:
                    bitmap[y][x] = colors[free_robot]
                else:
                    # Error color
                    bitmap[y][x] = np.array([1, 1, 1])

        dynamic_import["cv2"].imshow("Game", dynamic_import["cv2"].resize(
            bitmap, (960, 540)))
        dynamic_import["cv2"].waitKey(1)

        return bitmap

    def __str__(self) -> str:
        return "RoboticWarehouse"


class ActionSpace(gym.spaces.MultiDiscrete):
    def __init__(self, robots: int, categories: int):
        gym.spaces.MultiDiscrete.__init__(self, np.ones(robots) * 5)
