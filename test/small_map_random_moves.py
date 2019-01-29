import sys
import os
import time

# Hacky but tacky
sys.path.append(os.getcwd())

import robotic_warehouse.robotic_warehouse as rw

timestamp = time.time()
gym = rw.RoboticWarehouse(
    robots=4,
    capacity=1,
    spawn=10,
    spawn_rate=0.001,
    shelve_length=4,
    shelve_height=2,
    shelve_width=2,
    shelve_throughput=1,
    cross_throughput=5)
print("Setup Time: {}".format(time.time() - timestamp))

steps = 0
timestamp = time.time()
try:
    while True:
        gym.render()
        gym.step(gym.action_space.sample())
        steps += 1
        # time.sleep(0.1)
except KeyboardInterrupt:
    print("Number of steps: {}, average step per second: {}".format(
        steps, steps / (time.time() - timestamp)))

