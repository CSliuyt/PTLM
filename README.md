# [ICDE 2025] Improving Dataset Distillation with Two-fold Pseudo-Distribution Matching on Matching Capacity

#abstract
The ultimate goal of dataset distillation (DD) is to learn a compact synthetic dataset from the original, ensuring that models trained on both datasets demonstrate comparable generalization performance. As a cutting-edge approach to DD, distribution matching (DM) shows promising performance in addressing the challenge of model scalability. However, existing distillation approaches suffers from difficulty in dealing with data inherent feature and distribution shifts, and falls into a dilemma between efficiency and effectiveness. In this paper, we first examine the matching patterns in DM and show an insightful finding that a DM model should prioritize learning from samples in the original dataset that closely mirror its characteristics (similar samples) particularly when the image-per-class (IPC) count is low. Conversely, as the IPC increases, a DM model should integrate a broader spectrum of diverse samples (different samples) to capture more and extensive information from the original dataset. We call such a finding as capacity matching, and then we propose a Two-fold Pseudo-Distribution matching (namely Pseudo-Trajectory matching and Pseudo-Label Matching (PTLM)) to address feature and distribution shifting issues in DM. Specifically, we design (1) a Pseudo-Trajectory Matching (PTM) to address feature shift, and (2) Pseudo-Label Matching (PLM) to address distribution shift. Our proposal is a plug-and-play component for any DM-based method. Experimental results on multiple real-world datasets show the efficiency and effectiveness of the proposed method.

## Getting Started
1. Change the data paths and results paths in arguments/reproduce_xxxx.py
2. Perform the pre-training process
```
python pretrain.py -d cifar10 --reproduce
```
This will train multiple models from scratch and save their initial and ultimate state of dict.
3. Perform the condensation process using PTLM
```
python PTLM.py -d cifar10 --ipc 50 --factor 2 --reproduce
```

## Acknowledgement
Our code is built upon [IDC](https://github.com/snu-mllab/efficient-dataset-condensation) and  [DANCE](https://github.com/Hansong-Zhang/DANCE)












