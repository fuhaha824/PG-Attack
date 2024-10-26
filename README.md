# Precision-Guided Adversarial Attack

The official repository for Precision-Guided Adversarial Attack (PG-Attack).

**First-Place in the CVPR 2024 Workshop Challenge: Black-box Adversarial Attacks on Vision Foundation Models** 


Paper: *PG-Attack: A Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving* (https://arxiv.org/abs/2407.13111)

## Abstract
Vision foundation models are increasingly being employed in autonomous driving systems due to their advanced capabilities. However, these models are susceptible to adversarial attacks, posing significant risks to the reliability and safety of autonomous vehicles. Adversaries can exploit these vulnerabilities to manipulate the vehicle’s perception of its surroundings, leading to erroneous decisions and potentially catastrophic consequences. To address this challenge, we propose a novel Precision-Guided Adversarial Attack (PG-Attack) framework that combines two techniques: Precision Mask Perturbation Attack (PMP-Attack) and Deceptive Text Patch Attack (DTP-Attack). PMP-Attack precisely targets the attack region to minimize the overall perturbation while maximizing its impact on the target object’s representation in the model’s feature space. DTP-Attack introduces deceptive text patches that disrupt the model’s understanding of the scene, further enhancing the attack’s effectiveness. Our experiments demonstrate that PG-Attack successfully deceives a variety of advanced multi-modal large models, including GPT-4V, Qwen-VL, and imp-V1. Additionally, we won the First-Place in the CVPR 2024 Workshop Challenge: Black-box Adversarial Attacks on Vision Foundation Models.

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
