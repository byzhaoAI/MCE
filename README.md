# MCE
MCE: Towards a General Framework for Handling Missing Modalities under Imbalanced Missing Rates [[DOI]]() [[Arxiv]]()

Accepted by **Pattern Recognition** in 2025


## Overview
![MCE](./workflow.jpg)
**Abstract**: Multi-modal learning has made significant advances across diverse pattern recognition applications. However, handling missing modalities, especially under imbalanced missing rates, remains a major challenge. This imbalance triggers a vicious cycle: modalities with higher missing rates receive fewer updates, leading to inconsistent learning progress and representational degradation that further diminishes their contribution. Existing methods typically focus on global dataset-level balancing, often overlooking critical sample-level variations in modality utility and the underlying issue of degraded feature quality. We propose Modality Capability Enhancement (MCE) to tackle these limitations. MCE includes two synergistic components: i) Learning Capability Enhancement (LCE), which introduces multi-level factors to dynamically balance modality-specific learning progress, and ii) Representation Capability Enhancement (RCE), which improves feature semantics and robustness through subset prediction and cross-modal completion tasks. Comprehensive evaluations on four multi-modal benchmarks show that MCE consistently outperforms state-of-the-art methods under various missing configurations.


## Updates
- 2025/10/11 Create project. Support emotion recognition task on IEMOCAP.


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

## Acknowledgements
Thank for the project supported by [simple-bev](https://github.com/aharley/simple_bev), [RASSION](https://github.com/Jun-Jie-Shi/PASSION), [RedCore](https://github.com/sunjunaimer/RedCore)/[MMIN](https://github.com/AIM3-RUC/MMIN/tree/master) and [SMIL](https://github.com/deep-real/SMIL).
