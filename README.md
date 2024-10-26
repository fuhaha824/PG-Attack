# Precision-Guided Adversarial Attack

The official repository for Precision-Guided Adversarial Attack (PG-Attack).

**First-Place in the CVPR 2024 Workshop Challenge: Black-box Adversarial Attacks on Vision Foundation Models** 


Paper: *PG-Attack: A Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving* (https://arxiv.org/abs/2407.13111)

## Brief Introduction
Vision foundation models are widely used in autonomous driving, but they are vulnerable to adversarial attacks, risking vehicle safety. We propose PG-Attack, a framework combining Precision Mask Perturbation (PMP-Attack) and Deceptive Text Patch (DTP-Attack). PMP-Attack precisely targets regions to maximize impact, while DTP-Attack adds misleading text patches, confusing the model’s scene understanding. Our method effectively deceives advanced multi-modal models like GPT-4V, Qwen-VL, and imp-V1.

## Framework
<p align="left">
    <img src="./imgs/pipeline.png" width=83%\>
</p>


## Visualization
<p align="left">
    <img src="./imgs/vis.png" width=83%\>
</p>

## Citation
If you find our paper interesting or helpful to your research, please consider citing it, and feel free to contact fujy23@m.fudan.edu.cn if you have any questions.
```
@article{fu2024pgattack,
  title={PG-Attack: A Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving}, 
  author={Jiyuan Fu and Zhaoyu Chen and Kaixun Jiang and Haijing Guo and Shuyong Gao and Wenqiang Zhang},
  journal={arXiv preprint arXiv:2407.13111},
  year={2024}
}
```

## License

The project is **only free for academic research purposes** but has **no authorization for commerce**. 
