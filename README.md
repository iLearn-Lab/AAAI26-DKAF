<div align="center">
<h2 align="center">
     <b> Mitigating Endogenous Confirmation Bias in Noisy Label Learning for
     <br /> Vision-Language Models </b>
</h2>
<div>
<a target="_blank" href="https://openreview.net/profile?id=%7EFeiyang_Ning1">Feiyang&#160;Ning</a>,
<a target="_blank" href="https://scholar.google.com/citations?user=qVxhGWUAAAAJ&hl=zh-CN">Xinyang&#160;Chen</a><sup>&#9993</sup>,
</div>
School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), China&#160&#160&#160</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39641" target="_blank">
    <img src="https://img.shields.io/badge/Paper-AAAI-blue" alt="Paper AAAI"></a>
</div>
</div>

## :new: Updates
- [04/2026] :fire: Release code!
- [03/2026] :fire: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/39641) released!
- [01/2026] :fire: [Poster](https://github.com/Feiyang-Ning/DKAF/blob/main/poster/AAAI2026-DKAF.jpg) released!

## 📝 Abstract

> <div align="justify">
> Pretrained vision-language models (VLMs), especially CLIP, excel at adapting to downstream tasks through fine-tuning with sufficient high-quality labeled data. However, real-world training data often contains noisy labels, leading to significant performance degradation when models are naively fine-tuned on them. Existing noisy label learning methods for VLMs typically leverage the model's own pretrained knowledge, either via zero-shot predictions or vanilla self-training based on them, to identify and handle noisy samples. Crucially, these approaches blindly trust the VLM's pretrained knowledge, which can introduce <b>endogenous confirmation bias</b>: erroneous pretrained priors lead to incorrect noise detection, further amplifying the bias and corrupting the model. 
> <br><br>
> To overcome this limitation, we propose the <b>Debiased Knowledge Adaptation Framework (DKAF)</b>, which empowers the model to challenge and correct potentially flawed zero-shot predictions. DKAF operates in three progressive phases:
> <br>
> <b>(1) Clean Sample Selection</b>: We introduce a cross-modal collaborative pseudo-labeling to train a robust noisy label detector, explicitly mitigating confirmation bias by aggregating diverse signals beyond the model's initial zero-shot view.
> <br>
> <b>(2) Noisy Label Refinement</b>: For samples identified as noisy, we apply a dual-modal consistency strategy to selectively correct their labels, leveraging alignment between dominant and fused modalities to guide refinement while minimizing reliance on potentially biased internal knowledge.
> <br>
> <b>(3) Model Adaptation</b>: The model is progressively fine-tuned using the jointly curated dataset of selected clean samples and corrected noisy samples, promoting robust adaptation to the target task.
> <br><br>
> Extensive experiments on nine benchmark datasets (both synthetic and real-world noise) demonstrate that DKAF consistently outperforms state-of-the-art multimodal noisy label learning methods. Notably, under high-noise conditions, DKAF achieves average accuracy improvements of <b>3.08%</b>.
> </div>

## :pushpin: Poster
<div align="center">
  <a href="https://github.com/Feiyang-Ning/DKAF/blob/main/poster/AAAI2026-DKAF.jpg?raw=true" target="_blank">
    <img src="https://github.com/Feiyang-Ning/DKAF/blob/main/poster/AAAI2026-DKAF.jpg?raw=true" width="75%">
  </a>
</div>

## :rocket: Experiments
### 1. Environment Setup
Install the required dependencies:
```sh
pip install -r requirements.txt
```
### 2. Dataset Preparation
Create a [data](data) directory to store datasets and update the corresponding paths in the [config](config) files:
```sh
mkdir data
```
### 3. Pipeline
Run the following commands to experiment on CUB-200-2011 dataset with 60% symmetric noise:
```sh
# Phase 1&2: Clean Sample Selection + Noisy Label Refinement
python main_phase1_2.py --cfg "./config/PEFT/cub-200-2011.yaml" --noise_mode sym --noise_ratio 0.6 --seed 0
# Phase 3: Model Adaptation
python main_phase3.py --cfg "./config/FFT/cub-200-2011.yaml" --noise_mode sym --noise_ratio 0.6 --seed 0
```
### 3. More Examples
Reproduce other results from the paper with the following commands:
```sh
# 1. Experiment on synthetic dataset with 40% instance-dependent noise
python main_phase1_2.py --cfg "./config/PEFT/cub-200-2011.yaml" --noise_mode idn --noise_ratio 0.4 --seed 0
python main_phase3.py --cfg "./config/FFT/cub-200-2011.yaml" --noise_mode idn --noise_ratio 0.4 --seed 0

# 2. Experiment on real-world dataset
python main_real_phase1_2.py --cfg "./config/PEFT/cifar100N.yaml" --seed 0
python main_real_phase3.py --cfg "./config/FFT/cifar100N.yaml" --seed 0
```

## :star: Acknowledgements
We extend our sincere gratitude to the authors of [DeFT](https://github.com/HotanLee/DeFT) and [CoOp](https://github.com/KaiyangZhou/CoOp). Our codebase is built upon their outstanding work, and we highly appreciate their contribution to the research community.

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@inproceedings{ning2026mitigating,
  title={Mitigating Endogenous Confirmation Bias in Noisy Label Learning for Vision-Language Models},
  author={Ning, Feiyang and Chen, Xinyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={29},
  pages={24576--24584},
  year={2026}
}
```
