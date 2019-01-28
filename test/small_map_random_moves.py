import sys
import os
import time

# Hacky but tacky
sys.path.append(os.getcwd())

import robotic_warehouse.robotic_warehouse as rw

timestamp = time.time()
gym = rw.RoboticWarehouse(
    robots=10,
    capacity=1,
    speed=1,
    spawn=10,
    spawn_rate=0.1,
    shelve_length=2,
    shelve_height=10,
    shelve_width=10,
    shelve_throughput=1,
    cross_throughput=5)
print("Setup Time: {}".format(time.time() - timestamp))



while True:
    timestamp = time.time()
    # gym.render()
    gym.step(gym.action_space.sample())
    # time.sleep(0.1)
