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


class RoboticWarehouse(gym.Env):
    FREE = 0
    SHELF = 1
    PACKAGE_ID = 2
    PACKAGE = lambda x: (2, x)
    ROBOT_ID = 3
    ROBOT = lambda x: (3, x)
    DROP = 4

    UP = np.array([1, 0])
    DOWN = np.array([-1, 0])
    LEFT = np.array([0, -1])
    RIGHT = np.array([0, 1])

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
            periodicity_lower: int = 1,
            periodicity_upper: int = 10000
    ):  # How many places to spawn packages
        """ Number of packages a robot can hold. """
        self.capacity = capacity
        """ Remember this for environment resets. """
        self.inital_spawn = spawn
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
        """ Drop positions. (These must be initialized before setup env.) """
        # TODO: Make these depend on map
        self.drop_positions = [
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([0, 2])
        ]
        """ 
        For fast lookups if we ever have 
        A lot of drop positions and they are not dynamic. """
        self.drop_positions_set = set(
            map(lambda a: a.tostring(), self.drop_positions))
        """ Add drops to map for visually pleasing graphics ^^. """
        for y, x in self.drop_positions:
            self.map[y][x] = RoboticWarehouse.DROP
        """ To make sure same thing happends. """
        random.seed(seed)
        """ Package spawn positions. """
        self.package_spawn_positions = list(
            random.sample(self.shelve_positions, spawn))
        """ Make sure there is a periodicity pattern to the positions aswell (Something something can learn? :)) """
        self.package_spawn_times = [[
            0, random.randint(periodicity_lower, periodicity_upper), i
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

            if len(self.packages) == len(self.package_spawn_positions):
                logger.error(
                    "Cannot spawn more packages -- No Free positions -- Number Packes: {} -- Number Shelves: {}".
                    format(
                        len(self.packages), len(self.package_spawn_positions)))
                break

            identifier = np.random.randint(0, 2**32)
            while identifier in self.packages:
                identifier = random.randint(0, 2**32)

            y, x = random.choice(self.package_spawn_positions)
            while self.map[y][x] != RoboticWarehouse.SHELF:
                y, x = random.choice(self.package_spawn_positions)

            # Where to drop of package
            TO = random.choice(self.drop_positions)

            self.packages[identifier] = (np.array([y, x]), TO)
            self.map[y][x] = RoboticWarehouse.PACKAGE(identifier)
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
                A Robot is a List (Need to mutate top level data so cant be tuple)
                    0: 
                        0: Robot Y position
                        1: Robot X position
                    1: 
                        X: [Package]
                            Package: [From, To]
                                From: [Y, X]
                                To: [Y, X]
            """
            self.robots.append([np.array([y, x]), []])
            self.map[y][x] = RoboticWarehouse.ROBOT(robot)

    def reset(self) -> None:
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
                TO = random.choice(self.drop_positions)
                self.packages[identifier] = (np.array([y, x]), TO)
                self.map[y][x] = RoboticWarehouse.PACKAGE(identifier)

            package[0] = package[1]

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
        for r, action in enumerate(actions):
            reward = self.__actions[action](self.robots[r])

        return (self.robots, self.packages.values()), reward, False, None

    def __move_up(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.UP)

    def __move_down(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.DOWN)

    def __move_right(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.RIGHT)

    def __move_left(self, robot: list) -> int:
        return self.__move_direction(robot, RoboticWarehouse.LEFT)

    def __pickup_package(self, robot: list) -> int:
        """ Don't pick up anything if capacity is full. """
        if len(robot[1]) >= self.capacity:
            return
        """ Currently only picks in a grid.. maybe add diagonals?. """
        for package in [
                RoboticWarehouse.UP + robot[0],
                RoboticWarehouse.DOWN + robot[0],
                RoboticWarehouse.LEFT + robot[0],
                RoboticWarehouse.RIGHT + robot[0]
        ]:
            y, x = package
            if self.__within_map(y, x) and type(
                    self.map[y][x]
            ) == tuple and self.map[y][x][0] == RoboticWarehouse.PACKAGE_ID:
                """ 
                    Now add package to robot. 

                    Make sure
                        1: Package is also removed from map
                        2: Package is also removed from the free map
                """
                """ Add package to robot. """
                robot[1].append(self.packages[self.map[y][x][1]])
                """ Remove package from free packages. """
                del self.packages[self.map[y][x][1]]
                """ Remove package from map. """
                self.map[y][x] = RoboticWarehouse.SHELF

        return 0

    def __drop_package(self, robot: list) -> int:
        """ Don't try to drop anything if there is nothing. """
        if len(robot[1]) == 0:
            return 0

        for drop_position in [
                RoboticWarehouse.UP + robot[0],
                RoboticWarehouse.DOWN + robot[0],
                RoboticWarehouse.LEFT + robot[0],
                RoboticWarehouse.RIGHT + robot[0]
        ]:
            if drop_position.tostring() in self.drop_positions_set:
                score = len(robot[1])
                robot[1] = []
                return score

    def __move_direction(self, robot: list, direction: np.ndarray) -> int:
        oy, ox = robot[0]
        y, x = robot[0] + direction
        if self.__within_map(y, x) and self.map[y][x] == RoboticWarehouse.FREE:
            self.map[y][x], self.map[oy][ox] = self.map[oy][
                ox], RoboticWarehouse.FREE

            robot[0][0], robot[0][1] = y, x

        return 0

    def __within_map(self, y: int, x: int):
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

        free_robot = 5
        robot_with_package = 6
        colors = {
            RoboticWarehouse.FREE: np.array([.0, .0, .0]),
            RoboticWarehouse.SHELF: np.array([0.5, 0.2, 0.05]),
            RoboticWarehouse.PACKAGE_ID: np.array([0.0, 0.8, 0]),
            RoboticWarehouse.DROP: np.array([1.0, 0, 1.0]),
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

        ratio = self.map_width / self.map_height
        y_dim, x_dim = min(100 * self.map_height, 800), min(
            100 * self.map_width * ratio, 800 * ratio)

        if x_dim > y_dim and x_dim > 1000:
            y_dim *= (800 / x_dim)
            x_dim *= (800 / x_dim)

        dynamic_import["cv2"].imshow("Game", dynamic_import["cv2"].resize(
            bitmap, (int(x_dim), int(y_dim))))
        dynamic_import["cv2"].waitKey(1)

        return bitmap

    def __str__(self) -> str:
        return "RoboticWarehouse"


class ActionSpace(gym.spaces.MultiDiscrete):
    def __init__(self, robots: int, categories: int):
        gym.spaces.MultiDiscrete.__init__(self, np.ones(robots) * categories)
