# distributed-sensing

A repository to support the paper **Spatial Scheduling of Informative Meetings for Multi-Agent Persistent Coverage**.

Videos:  
[![click to play](https://img.youtube.com/vi/M5Fp8WsmLno/0.jpg)](https://www.youtube.com/watch?v=M5Fp8WsmLno)
[![click to play](https://img.youtube.com/vi/gLdqK2m3COo/0.jpg)](https://www.youtube.com/watch?v=gLdqK2m3COo)

Paper citation:
```
@Article{9001230,
  author={R. N. {Haksar} and S. {Trimpe} and M. {Schwager}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Spatial Scheduling of Informative Meetings for Multi-Agent Persistent Coverage}, 
  year={2020},
  volume={5},
  number={2},
  pages={3027-3034},}
```

## Requirements:
- Developed with Python 3.6
- Requires the `numpy` and `networkx` packages
- Requires the [simulators](https://github.com/rhaksar/simulators) repository 

## Directories:
- `framework`: Implementation of scheduling and planning framework. 

## Files:
- `Baseline.py`: Implementation of two baseline algorithms to compare against the framework. 
- `Benchmark.py`: Run many simulations of the framework to evaluate perofrmance. 
- `Meetings.py`: Run a single simulation of the framework. 

## realworld
data collected from realworld experiments, the data in the folder env15fov3rho2345fire3seed11 is used

## mobicom plot
- `mobi_statis1.ipynb`: for simulation performance
- `mobi_statis2.ipynb`: for system robustness
- `mobi_statis3.ipynb`: for real world experiments

## mobicom_eva
pkl data used in the mobicom2023 paper

## mobicom_vis
plot results in teh mobicom2023 paper


## other_methods
- `LevyFlight.py`: both levyflightplus and levyflight are in this file
- `madqn.py`: ddrl is in this file

To facilitate ddrl, class `UAV` in `frameworks/uav.py` is not enough for use. To this end, a derived class `UAV_ddrl` is added according to original ddrl implementation.

When loading the pretrained model, the image_dim specified when training should be the same as `image_size` when testing. (currently, two models are all trained under image_dim=(3,3))