# MCE
MCE: Towards a General Framework for Handling Missing Modalities under Imbalanced Missing Rates [[DOI]]() [[Arxiv]]()

Accepted by **Pattern Recognition** in 2025


## Overview
![MCE](./workflow.jpg)
**Abstract**: Multi-modal learning has made significant advances across diverse pattern recognition applications. However, handling missing modalities, especially under imbalanced missing rates, remains a major challenge. This imbalance triggers a vicious cycle: modalities with higher missing rates receive fewer updates, leading to inconsistent learning progress and representational degradation that further diminishes their contribution. Existing methods typically focus on global dataset-level balancing, often overlooking critical sample-level variations in modality utility and the underlying issue of degraded feature quality. We propose Modality Capability Enhancement (MCE) to tackle these limitations. MCE includes two synergistic components: i) Learning Capability Enhancement (LCE), which introduces multi-level factors to dynamically balance modality-specific learning progress, and ii) Representation Capability Enhancement (RCE), which improves feature semantics and robustness through subset prediction and cross-modal completion tasks. Comprehensive evaluations on four multi-modal benchmarks show that MCE consistently outperforms state-of-the-art methods under various missing configurations.


## Updates
- 2025/10/11 Create project. Support emotion recognition task on IEMOCAP, digit recognition task on AudiovisionMNIST.


## Dataset Support
  - [x] nuScenes
  - [ ] BraTS2020
  - [x] IEMOCAP
  - [x] AudiovisionMNIST

## Quick Start
### 1. nuScenes
##### Dataset setup
```
cd scene_seg
mkdir nuScenes
```
First, you need to download raw data from [nuScenes](https://www.nuscenes.org/) to the created path `/nuScenes`.

##### Pretrained unimodal setup
Coming soon ...

##### Install and Train
Please refer to the [scene_seg/README.md](./scene_seg/README.md) for detailed documentations.


### 2. IEMOCAP
##### Dataset setup
```
cd emotion_recog
mkdir inputs
mkdir inputs/IEMOCAP
```
1. Download the IEMOCAP extracted features `IEMOCAP_features.zip` from [GoogleDrive](https://drive.google.com/drive/folders/1OPiw4XFTzTnoKxT14jtSif5XVN8Qe7l4?usp=sharing) to the created path `/inputs/IEMOCAP/`.
2. unzip `IEMOCAP_features.zip`
3. Set the path to the features in the json files under folder: data/config/ (Only check)

##### Pretrained unimodal setup
```
mkdir unimodal_checkpoints
```
Download desired pretrained encoder&decoder from [GoogleDrive](https://drive.google.com/drive/folders/1AQmQnV-wW-aM6btdHlb4OPr4hrmtrlTW?usp=sharing) to the created path `/unimodal_checkpoints`.

##### Install
```
cd emotion_recog
conda env create -f environment.yml
```

##### Train
Activate the environment `mce_iemocap` first, then
```
sh IEMOCAP_MCE.sh
```

### 3. AudiovisionMNIST
##### Dataset setup
```
cd digit_recog
mkdir soundmnist
```
- Download the data from [GoogleDrive](https://drive.google.com/file/d/1JTS--8d_BxzZfhQfSAAYeYTjCdUbJyuD/view?usp=sharing) and extracted it to `/soundmnist`

##### Pretrained unimodal setup
```
mkdir unimodal_ckpts
mkdir unimodal_ckpts/image
mkdir unimodal_ckpts/sound
```
Download desired pretrained encoder&decoder from [GoogleDrive](https://drive.google.com/drive/folders/1AQmQnV-wW-aM6btdHlb4OPr4hrmtrlTW?usp=sharing) to the created path `/unimodal_checkpoints`.

##### Install
```
cd digit_recog
conda env create -f environment.yml
```

##### Train
Activate the environment `audiovision` first, then
```
python train_mce.py
```

## Citation
If you are using our project for your research, please cite the following paper:

```
Coming soon...
```

or

```
Coming soon...
```

1. If you are using `scene_seg`, please also cite:
- **nuScenes: A multimodal dataset for autonomous driving**
```
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11621--11631},
  year={2020}
}
```
- **Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?**
```
@inproceedings{harley2023simple,
  title={Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?},
  author={Harley, Adam W and Fang, Zhaoyuan and Li, Jie and Ambrus, Rares and Fragkiadaki, Katerina},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2759--2765},
  year={2023},
  organization={IEEE}
}
```

2. If you are using `emotion_recog`, please also cite:
- **RedCore: Relative advantage aware cross-modal representation learning for missing modalities with imbalanced missing rates**
```
@inproceedings{sun2024redcore,
  title={RedCore: Relative advantage aware cross-modal representation learning for missing modalities with imbalanced missing rates},
  author={Sun, Jun and Zhang, Xinxin and Han, Shoukang and Ruan, Yu-Ping and Li, Taihao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={15173--15182},
  year={2024}
}
```

3. If you are using `digit_recog`, please also cite:
- **SMIL: Multimodal Learning with Severely Missing Modality**
```
@inproceedings{ma2021smil,
  title={Smil: Multimodal learning with severely missing modality},
  author={Ma, Mengmeng and Ren, Jian and Zhao, Long and Tulyakov, Sergey and Wu, Cathy and Peng, Xi},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={3},
  pages={2302--2310},
  year={2021}
}
```

## Acknowledgements
Thank for the project supported by 
- scene segmentation: [SimpleBEV](https://github.com/aharley/simple_bev)
- medical image segmentation: [RASSION](https://github.com/Jun-Jie-Shi/PASSION)
- emotion recognition: [RedCore](https://github.com/sunjunaimer/RedCore)/[MMIN](https://github.com/AIM3-RUC/MMIN/tree/master)
- digit recognition: [SMIL](https://github.com/deep-real/SMIL)
