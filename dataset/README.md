# Routes and scenarios
The routes (`./routes`) and scenarios (`./scenarios`) are borrowed from the [TransFuser repository](https://github.com/autonomousvision/transfuser/).
The core routes (`./routes/core`) consists of the long routes (Town01-Town06) from the [official CARLA leaderboard](https://github.com/carla-simulator/leaderboard/tree/master/data) plus the short and tiny routes which appear in [TransFuser repository CVPR2021 branch](https://github.com/autonomousvision/transfuser/tree/cvpr2021/leaderboard/data). 
The corresponding scenarios can be found in `./scenarios/all_towns_traffic_scenarios_public.json` which is also from the official leaderboard.
These routes are also used by [InterFuser](https://github.com/opendilab/InterFuser/tree/main/leaderboard/data) for their training.

TransFuser also provides additional routes 3500 training routes in [Town01-Town07 and Town10HD](https://carla.readthedocs.io/en/latest/core_map/#carla-maps) 
for [Scenarios 1, 3, 4, 7, 8, 9, and 10](https://leaderboard.carla.org/scenarios/), along with lane changes (left-left, left-right, right-left, right-right).
Their [route and scenario generation scripts](https://github.com/autonomousvision/transfuser/tree/2022/tools) which we borrowed can be found in `./transfuser/tools`.

[TCP](https://github.com/OpenPerceptionX/TCP/tree/main/leaderboard/data/TCP_training_routes) uses a different set of training routes for training and validation, which we also provide in `./tcp`.

## Longest6 benchmark
Borrowed from the [TransFuser Longest6 benchmark](https://github.com/autonomousvision/transfuser/tree/2022/leaderboard/data/longest6), under `./benchmark`.

The Longest6 benchmark consists of 36 routes with an average route length of 1.5km, which is similar to the average route length of the official leaderboard (~1.7km). During evaluation, we ensure a high density of dynamic agents by spawning vehicles at every possible spawn point permitted by the CARLA simulator. Following the [NEAT evaluation benchmark](https://github.com/autonomousvision/neat/blob/main/leaderboard/data/evaluation_routes/eval_routes_weathers.xml), each route has a unique environmental condition obtained by combining one of 6 weather conditions (Cloudy, Wet, MidRain, WetCloudy, HardRain, SoftRain) with one of 6 daylight conditions (Night, Twilight, Dawn, Morning, Noon, Sunset).
