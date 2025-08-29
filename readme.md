# README

## 0. conda init

### Create Conda env
```bash
conda env create -n RISE_robo -f environment.yml
conda activate RISE_robo
```
**torch 2.3.1 / torchvision 0.18.1 / torchaudio 2.3.1**

**mujoco 3.3.5 / robosuite 1.5.1**

**MinkowskiEngine 0.5.4**

---

## 1. dataset get
```bash
bash dataset.sh
```
**可配置项**
- cam_id
- cam_names
- cam_height / cam_width

**device control robot - keyboard**
- "space" - toggle gripper (open/close)
- "up-right-down-left" - move horizontally in x-y plane
- "o-p / y-h / e-r" - yaw / pitch / roll
- "q" - save epoch
---

## 2. train
```bash
bash command_train.sh
```
**可配置项**
- target_obj
- cam_ids: use one cam
- aug: pic / data aug
- batch_size
- num_epochs: train_epoch num
- save_epochs: Save a weight file once
- num_workers
- ckpt_dir: {root directory} + ckpt_dir
---

## 3. eval
```bash
bash command_eval.sh
```
**可配置项**
- ckpt_dir: {root directory} + ckpt_dir
- target_obj
- cam_names: use one cam
- cam_height / cam_width
- voxel_size
- num_action
---

## 4. 目录结构（示例）
```
project_root/
├─ assets/
├─ calib/
├─ checkpoints/
├─ data/
├─ dataset/
├─ dep/
├─ model/
├─ policy/
├─ utils/
├─ eval.py
├─ train.py
├─ record_episodes.py
├─ readme.md
├─ LICENSE
├─ environment.yml
├─ dataset.sh
├─ command_eval.sh
└─ command_train.sh
```
---
## 5. Upstream & License

本项目基于 **RISE** 进行二次开发（IROS 2024）：
- Upstream: https://github.com/rise-policy/RISE
- 上游许可：CC BY-NC-SA 4.0
- 本仓库中直接来源或修改自 RISE 的代码与配置，遵循 CC BY-NC-SA 4.0；其余自研部分亦以兼容方式发布（如无特殊声明，默认同上游）。

### Acknowledgements
- Diffusion 模块参考 RISE 上游（其又参考 Diffusion Policy / MIT），Transformer 模块参考 ACT / DETR（Apache-2.0），稀疏 3D 编码参考 MinkowskiEngine 示例（MIT）。具体来源与许可请见上游 README 与 LICENSE。

### Citation
如在学术工作中使用本仓库，请同时引用 RISE：
