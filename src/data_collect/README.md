## Data Collector

This directory contains scripts used for data collection.
It is based on [leaderboard_evaluator.py](https://github.com/carla-simulator/leaderboard/blob/master/leaderboard/leaderboard_evaluator.py)
in the official [leaderboard repository](https://github.com/carla-simulator/leaderboard) with some minor changes to adapt it for data collection purposes.

These changes are:

- adding `'sensor.camera.semantic_segmentation'`, `'sensor.camera.depth'`, `'sensor.lidar.ray_cast_semantic'`, to available sensors,
- terminating scenarios upon collision or running red lights,
- allowing the agent to have access to the `ego_vehicle`
- adding the option to set the weather
- don't add noise to gnss during data collection

These changes are made following the example set by [TCP](https://github.com/OpenPerceptionX/TCP). 
Rather than modifying a copy of the official leaderboard like TCP, we extracted the changes made into this directory without 
touching the official leaderboard implementation, and only modified the `leaderboard_evaluator.py` 
(renaming it to `data_collector.py`) to suit our purpose.
