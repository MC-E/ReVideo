# ReVideo: Remake a Video with Motion and Content Control
[Chong Mou](https://scholar.google.com/citations?user=SYQoDk0AAAAJ&hl=zh-CN),
[Mingdeng Cao](https://scholar.google.com/citations?user=EcS0L5sAAAAJ&hl=en),
[Xintao Wang](https://xinntao.github.io/),
[Zhaoyang Zhang](https://zzyfd.github.io/),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ),
[Jian Zhang](https://jianzhang.tech/)

[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://mc-e.github.io/project/ReVideo/)
[![arXiv](https://img.shields.io/badge/ArXiv-2405.13865-brightgreen)](https://arxiv.org/abs/2405.13865)

---
## Introduction
ReVideo aims to solve the problem of local video editing. The editing target includes visual content and motion trajectory modifications.
<p align="center">
  <img src="asserts/teaser.jpg">
</p>

## 📰 **New Features/Updates**
- [2024/09/25] ReVideo is accepted by NeurIPS 2024.
- [2024/06/26] We release the code of ReVideo.
- [2024/05/26] **Long video editing plan**: We are collaborating with [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) team to replace SVD with Sora framework, making ReVideo suitable for long video editing. Here are some preliminary results. This initial combination is still limited in quality for long videos. In the future, we will continue to cooperate and launch high-quality long video editing models.
<table class="center">
<tr>
  <td style="text-align:center;"><b>Generated by Open-Sora</b></td>
  <td style="text-align:center;"><b>Editing Result</b></td>
</tr>
<tr>
  <td><video src="https://github.com/MC-E/ReVideo/assets/54032224/81241556-0f1b-438e-ba90-094d7cc0eded" autoplay></td>
  <td><video src="https://github.com/MC-E/ReVideo/assets/54032224/474b3620-f156-4d30-a473-cbbcc615f56c" autoplay></td>
</tr>
</table>
- [2024/05/23] Paper and project page of **ReVideo** are available.

## ✏️ Todo
- [x] Code will be open-sourced in June

## 🔥🔥🔥 Main Features
### Change content & Customize motion trajectoy
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Editing Result</b></td>
</tr>
<tr>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/222f35da-7396-4989-a3c3-9ab4a2e5fa98" autoplay></td>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/c128f1d7-30e4-49e7-b6b7-9d5f428ff882" autoplay></td>
</tr>
</table>

### Change content & Keep motion trajectoy
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Editing Result</b></td>
</tr>
<tr>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/d25dce6a-88cf-45ad-9177-76df9fffe819" autoplay></td>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/06c8f19d-4569-417f-a4a3-1782a09404db" autoplay></td>
</tr>
</table>

### Keep content & Customize motion trajectoy
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Editing Result</b></td>
</tr>
<tr>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/490b4e9b-c1af-4f87-83de-c6b27f4a925b" autoplay></td>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/93f77c7b-23a8-4b1e-8e6d-1abf57fd1130" autoplay></td>
</tr>
</table>

### Multi-area Editing
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Editing Result</b></td>
</tr>
<tr>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/339263b6-ea97-4c43-8617-b40459b1973c" autoplay></td>
  <td><video src="https://github.com/MC-E/DragonDiffusion/assets/54032224/7a005b3a-ff3e-492c-9643-0fd921b0b53e" autoplay></td>
</tr>
</table>

## 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

## ⏬ Download Models 
All models will be automatically downloaded. You can also choose to download manually from this [url](https://huggingface.co/Adapter/ReVideo).

**Since our ReVideo is trained based on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid), the usage guidelines for the model should follow the Stable Video Diffusion's [NC-COMMUNITY LICENSE](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE)!**

## 💻 How to Test
You can download the testset from  [https://huggingface.co/Adapter/ReVideo](https://huggingface.co/Adapter/ReVideo).
Inference requires at least `20GB` of GPU memory for editing a `768x1344` video.  

```bash
bash configs/examples/constant_motion/head6.sh
```

### Description of the same parameters
```bash
--s_h  # The abscissa of the top left corner of the editing region
--e_h # The abscissa of the lower right corner of the editing region
--s_w # The ordinate of the top left corner of the editing region
--e_w # The ordinate of the lower right corner of the editing region
--ps_h # The abscissa of the start point
--pe_h # The abscissa of the end point
--ps_w # The ordinate of the start point
--pe_w # The ordinate of the end point
--x_bias_all # Horizontal offset of reciprocating motion
--y_bias_all # Vertical offset of reciprocating motion
```

## Related Works
<p>
[1] <a href="https://pika.art/">https://pika.art/</a>
</p>
<p>
[2] <a href="https://arxiv.org/abs/2308.08089">DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory</a>
</p>
<p>
[3] <a href="https://arxiv.org/abs/2403.07420">
    DragAnything: Motion Control for Anything using Entity Representation</a>
</p>
<p>
[4] <a href="https://arxiv.org/abs/2403.14468/">AnyV2V: A Plug-and-Play Framework For Any Video-to-Video Editing Tasks</a>
</p>

# 🤗 Acknowledgements
We appreciate the releasing code of [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).
