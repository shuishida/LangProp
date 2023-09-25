# TCP - Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline

![teaser](../../../3rdparty/TCP/assets/teaser_.png)

> Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline  
> Penghao Wu*, [Xiaosong Jia*](https://jiaxiaosong1002.github.io/), Li Chen*, [Junchi Yan](https://thinklab.sjtu.edu.cn/), [Hongyang Li](https://lihongyang.info/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/)    
>  - [arXiv Paper](https://arxiv.org/abs/2206.08129), NeurIPS 2022
>  - [Blog in Chinese](https://zhuanlan.zhihu.com/p/532665469)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/trajectory-guided-control-prediction-for-end/autonomous-driving-on-carla-leaderboard)](https://paperswithcode.com/sota/autonomous-driving-on-carla-leaderboard?p=trajectory-guided-control-prediction-for-end)

This repository contains the code for the paper [Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline](https://arxiv.org/abs/2206.08129).


TCP is a simple unified framework to combine trajectory and control prediction for end-to-end autonomous driving.  By time of release in June 17 2022, our method achieves new state-of-the-art on [CARLA AD Leaderboard](https://leaderboard.carla.org/leaderboard/), in which we rank the **first** in terms of the Driving Score and Infraction Penalty using only a single camera as input. 


## Setup
1. Follow `README.md` in `CarlaCode` root directory. It will perform the following steps:
   - Download [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1).

2. Run the setup script from this directory.
   ```
   bash setup.sh
   ```
   This will perform the following steps:
   - Create a [conda](https://docs.conda.io/en/latest/miniconda.html) environment.
   - Install required carla packages.
   - Add Carla-related paths to $PYTHONPATH

## Dataset

Download our dataset through [GoogleDrive](https://drive.google.com/file/d/1A6k0KpVHs5eSaqunzbRQR-p0b-915O9R/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/11xBZwAWQ3WxQXecuuPoexQ) (提取码 8174). The total size of our dataset is aroung 115G, make sure you have enough space.

## Pretrained weights
Pretrained weights are provided as `TCP.ckpt`, and the resulting statistics in `results_TCP.json`, downloaded from [this issue](https://github.com/OpenPerceptionX/TCP/issues/23).

## Training
First, set the dataset path in ``TCP/config.py``.
Training:
```
python TCP/train.py --gpus NUM_OF_GPUS
```

## Data Generation
First, launch the carla server,
```
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```
Set the carla path, routes file, scenario file, and data path for data generation in ``leaderboard/scripts/data_collection.sh``.

Start data collection

```
sh leaderboard/scripts/data_collection.sh
```
After the data collecting process, run `tools/filter_data.py` and `tools/gen_data.py` to filter out invalid data and pack the data for training.

## Evaluation
First, launch the carla server,
```
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```
Set the carla path, routes file, scenario file, model ckpt, and data path for evaluation in ``leaderboard/scripts/run_evaluation.sh``.

Start the evaluation

```
sh leaderboard/scripts/run_evaluation.sh
```

## Citation

If you find our repo or our paper useful, please use the following citation:

```
@article{wu2022trajectoryguided,
 title={Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline}, 
 author={Penghao Wu and Xiaosong Jia and Li Chen and Junchi Yan and Hongyang Li and Yu Qiao},
 journal={arXiv preprint arXiv:2206.08129},
 year={2022},
}
```

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements

Our code is based on several repositories:
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)

