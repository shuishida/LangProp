# Comparison of baselines

We compare the following third party baselines, which are cloned into the repository as git submodules at [../../3rdparty](../../3rdparty). 
In addition, we have included the core agent implementation in this directory to have a standardized comparison using the same leaderboard evaluator.
The evaluation scripts are under the [../../scripts/eval_expert](../../scripts/eval_expert) directory.
For more information on running the baselines, refer to the `README.md` files under the subdirectory corresponding to your method of interest.

- [Carla Garage: Hidden Biases of End-to-End Driving Models
](https://github.com/autonomousvision/carla_garage)
- [InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer
](https://github.com/opendilab/InterFuser)
- [TCP - Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline
](https://github.com/OpenPerceptionX/TCP)
- [MILE: Model-Based Imitation Learning for Urban Driving](https://github.com/wayveai/mile)
- [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving
](https://github.com/autonomousvision/transfuser)
- [CARLA-Roach
](https://github.com/zhejz/carla-roach)


| Method               | Roach                                              | Mile             | TransFuser        | TCP              | InterFuser        | TF++              |
|----------------------|----------------------------------------------------|------------------|-------------------|------------------|-------------------|-------------------|
| Expert               | Roach                                              | Roach            | -                 | Roach            | -                 | -                 |
| Training of expert   | RL with privileged <br/>information as observation | Pretrained Roach | Rule-based expert | Pretrained Roach | Rule-based expert | Rule-based expert |


## [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](https://github.com/autonomousvision/transfuser)
#### TL;DR TransFuser integrates image and LiDAR inputs by applying transformers at multiple resolutions to fuse perspective view and bird’s eye view feature maps. They also propose the Longest6 benchmark for offline evaluation on CARLA.

## [TCP - Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline](https://github.com/OpenPerceptionX/TCP)
#### TL;DR Driving models typically predict either 1) waypoints of a trajectory or 2) direct control (velocity). TCP combines the two approaches, keeping the benefit of multi-step predictions from 1 while making the control problem end-to-end trainable like 2.

TCP has two prediction branches - a trajectory branch which predicts K future waypoints, 
and a control branch which makes multi-step control predictions (e.g. velocity). 
The control branch receives guidance from the trajectory branch at each time step. 
The outputs from two branches are fused to achieve complementary advantages.

## [InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://github.com/opendilab/InterFuser)
TL;DR InterFuser processes and fuses information from multimodal multi-view sensors for comprehensive scene understanding and adversarial event detection.
