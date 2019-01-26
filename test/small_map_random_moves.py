import sys
import os
import time

# Hacky but tacky
sys.path.append(os.getcwd())

import robotic_warehouse.robotic_warehouse as rw

gym = rw.RoboticWarehouse(
    robots=1,
    capacity=1,
    speed=1,
    spawn=10,
    spawn_rate=1,
    shelve_length=2,
    shelve_height=5,
    shelve_width=10,
    shelve_throughput=1,
    cross_throughput=5)



while True:
    gym.render()
    time.sleep(1)
