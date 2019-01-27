import sys
import os
import time

# Hacky but tacky
sys.path.append(os.getcwd())

import robotic_warehouse.robotic_warehouse as rw

timestamp = time.time()
gym = rw.RoboticWarehouse(
    robots=1,
    capacity=1,
    speed=1,
    spawn=10,
    spawn_rate=1,
    shelve_length=5,
    shelve_height=2,
    shelve_width=2,
    shelve_throughput=1,
    cross_throughput=5)
print("Setup Time: {}".format(time.time() - timestamp))



while True:
    timestamp = time.time()
    gym.render()
    print("Render Time: {}".format(time.time() - timestamp))
    time.sleep(0.1)
