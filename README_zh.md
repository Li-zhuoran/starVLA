# StarVLA: 用于视觉-语言-动作模型开发的积木式代码库

![更新](https://img.shields.io/badge/更新-脚本已修复%20%7C%20打包更流畅-red?style=for-the-badge)

**[2025/11/12]** 我们现在支持[Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model)作为资源受限开发的小型VLM。StarVLA现在可以在单个A100 GPU上运行。更多详情请参见[🚀使用较小的VLM训练](#train-smaller-vlm)部分。

**[2025/10/30]:** 我们发布了LIBERO训练与评估README。结果非常有前景。更多详情请见[examples/LIBERO](examples/LIBERO)。

**[2025/10/25]:** 我们修复了几个脚本链接，现在一切更加顺畅。感谢社区的反馈。

---

StarVLA是一个模块化且灵活的代码库，用于开发从视觉-语言模型(VLM)到视觉-语言-动作(VLA)模型的转换。
在StarVLA（也是"start VLA"的双关语）中，每个功能组件（模型、数据、训练器、配置、评估等）都遵循自顶向下、直观分离以及高内聚低耦合的原则，实现了即插即用的设计、快速原型设计和独立调试。

![](assets/Framworks.png)
*实线边框表示已支持的模块；无边框的模块即将推出。

## 🔥 主要特性

<details open>
<summary><b>多种VLA框架</b></summary>

- [x] **Qwen-FAST**: 使用Qwen2.5-VL-3B配合快速分词器，根据视觉和语言输入自回归地生成离散动作标记（类似于π₀-fast）。
- [x] **Qwen-OFT**: 将Qwen2.5-VL-3B与MLP动作头结合，从预定义特殊动作标记的隐藏状态中并行解码连续动作（类似于OpenVLA-OFT/EO）。
- [x] **Qwen-PI**: 将流匹配(FM)动作专家与Qwen2.5-VL-3B集成，采用基于扩散的方法预测连续动作（类似于π₀）。
- [x] **Qwen-GR00T**: 实现双系统VLA架构，其中Qwen2.5-VL-3B作为System2进行高级视觉-语言推理，而流匹配模块作为System1进行快速动作预测（类似于GR00T）。

<p align="center">
  <img src="assets/starvla_simpleEnv.png" alt="SimplerEnv 模块" width="95%">
</p>

<p align="center">
  <img src="assets/starvla_LIBERO.png" alt="LIBERO 模块" width="84%">
</p>

有关动态更新，请参阅我们的[🍀 Overleaf](https://www.overleaf.com/read/qqtwrnprctkf#d5bdce)，其中持续展示我们的实时实验结果。

### 📈 模型库

我们发布了一系列预训练模型和检查点，以促进复现和下游应用。

#### ✅ 可用检查点

| 模型 | 描述 | WindowX | 链接 |
|-------|-------------|------|------|
| **Qwen2.5-VL-3B-Action** | 在Qwen2.5-VL基础上添加动作标记 | - | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) |
| **Qwen3-VL-4B-Action** | 在Qwen3-VL基础上添加动作标记 | - | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action) |
| **QWen2.5-FAST-Bridge-RT-1** | QwenVL + 快速分词器 | 58.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FAST-Bridge-RT-1) |
| **QWen2.5-OFT-Bridge-RT-1** | QwenVL + OFT 动作回归 | 41.8 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-OFT-Bridge-RT-1) |
| **QWen2.5-PI-Bridge-RT-1** | QwenVL + 流匹配专家 | 62.5 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FM-Bridge-RT-1) |
| **QWen2.5-GR00T-Bridge-RT-1** | QwenVL + GR00T N1.5 动作头 | 63.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-PI-Bridge-RT-1) |
| **QWen-GR00T-Bridge** | QwenVL + GR00T N1.5 动作头 | 71.4 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-GR00T-Bridge) |
| **QWen3VL-OFT-Bridge-RT-1** | Qwen3VL + OFT 动作回归 | 42.7 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-OFT-Bridge-RT-1) |
| **QWen3VL-GR00T-Bridge-RT-1** | Qwen3VL + GR00T N1.5 动作头 | 65.3 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) |

---
</details>

<details close>
<summary><b>多种仿真基准测试</b></summary>

- [x] **SimplerEnV**
- [x] **LIBERO**
- [ ] **Robocasa**
- [ ] **RLBench**
- [ ] **RoboTwin**
- [ ] **BEHAVIOR**

</details>

<details close>
<summary><b>多种训练策略</b></summary>

* [x] 单一模仿学习
* [x] 多模态多任务协同训练
* [ ] 强化学习适应

</details>

---

## 🌟 StarVLA如何让模型开发再次变得像积木一样？

👇 StarVLA通过以下设计实现"积木式"开发：

<a id="model"></a>
<details close>
<summary><b>1. 子模块冒烟测试</b></summary>

StarVLA强调模块化模型设计。每个主要框架文件都可以独立运行，以便快速调试和冒烟测试您的代码。例如：

```bash
# 模型
python starVLA/model/framework/QwenOFT.py --config_yaml starvla_cotrain_oxe.yaml
# 数据加载器
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml
```

注意：`starVLA/model/framework/yourframework.py`是模型的单一外部API接口；它应该与论文中的框架图结构同构。
</details>

<a id="data"></a>
<details close>
<summary><b>2. 明确的模型边界</b></summary>

StarVLA遵循自顶向下分解和高内聚低耦合原则。

例如：
- 数据加载器
  - 仅返回原始的、与模型无关的字典；不进行模型特定的预处理（如分词器、图像编码）。
  - 单个样本应包括（根据需要添加/删除）：
    - image: list[PIL.Image] | np.ndarray
    - lang: str
    - action: np.ndarray[T, action_dim]
    - state: Optional[np.ndarray[..., state_dim]]

`framework.forward()`和`framework.predict_action()`都直接操作原始输入，使训练/测试边界明确且易于修改。
</details>

<a id="config"></a>
<details close>
<summary><b>3. 灵活的配置系统</b></summary>

StarVLA使用单一全局配置对象。
参数主要通过可扩展的字典传递，允许覆盖和受控冗余。

</details>

🧪 *为了自我测试和迭代StarVLA的可用性，我们重新实现了几种代表性的VLA框架。我们进行了beta测试：内部开发人员可以在半天内（至少3小时内）搭建新的VLA框架，新用户可以在一天内构建他们的第一个自定义VLA框架。每项设计的更多见解可在[assets/intro_v1.md](assets/intro_v1.md)中找到。*

---

## 🚀 快速开始

<details close>
<summary><b>🛠 环境设置</b></summary>

```bash
# 克隆仓库
git clone https://github.com/starVLA/starVLA

# 创建conda环境
conda create -n starVLA python=3.10 -y
conda activate starVLA

# 安装依赖
pip install -r requirements.txt

# 安装FlashAttention2
pip install flash-attn --no-build-isolation

# 安装starVLA
pip install -e .
```

⚠️ **常见问题**
flash-attn安装可能比较困难，因为它必须与系统的CUDA工具包(nvcc)和PyTorch版本匹配。`--no-build-isolation`标志可以解决大多数问题，但在较新系统上可能需要手动选择兼容的flash-attn版本。确保CUDA驱动/工具包和torch版本一致。检查您的环境：

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

如果问题仍然存在，请选择与您的版本(CUDA和torch)匹配的flash-attn发行版，或者使用chatGPT搜索功能帮助解决问题。

</details>

<details close>
<summary><b>👀 快速检查StarVLA</b></summary>

```bash
# 使用假数据检查框架
python starVLA/model/framework/QwenGR00T.py
```

您应该下载`./playground/Pretrained_models/Qwen3-VL-4B-Instruct`。它应该成功构建并`print(model)`。您也可以调用`model.forward(fake_data)`并通过`model.predict_action(fake_data)`获得未归一化的动作。

</details>

<details close>
<summary><b>🧪 评估现有模型</b></summary>

我们还提供了一个并行评估脚本：

```bash
check_pt=StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
bash examples/SimplerEnv/star_bridge_parall_eval.sh ${check_pt}
```

运行前请下载[Qwen3VL-GR00T-Bridge-RT-1](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1)并按照[SimpplerEnv](https://simpler-env.github.io/)准备Python环境。直接在[star_bridge_parall_eval.sh](file:///home/lzr/robot/starVLA/examples/SimplerEnv/star_bridge_parall_eval.sh)顶部编辑这些变量。

如果您不想进行并行测试，请运行：

```bash
# 终端1
bash ./examples/SimplerEnv/start_server.sh
# 终端2
bash ./examples/SimplerEnv/start_simpler_env.sh
```

---

⚠️ **常见问题**
在NVIDIA A100上测试SimplerEnv时，您可能会遇到以下错误：
`libvulkan.so.1: cannot open shared object file: No such file or directory`
您可以参考此链接修复：[安装指南–Vulkan部分](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

运行策略服务器但出现`NotImplementedError:Framework QwenGR00T is not implemented`时，您可能需要`python QwenGR00T.py`来检查您的环境。

</details>

<details close>
<summary><b>🚀 训练您自己的模型</b></summary>

我们的训练管道遵循[InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1/examples/SimplerEnv)。

步骤：
1) 准备LeRobot格式的OXE数据集，包括[modality.json](file:///home/lzr/robot/starVLA/examples/LIBERO/train_files/modality.json)。请参考[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SimplerEnv)。
2) 将您的数据集路径添加到`config.yaml`：
    ```yaml
    datasets:
      vla_data:
        dataset_py: lerobot_datasets
        data_root_dir: playground/Datasets/OXE_LEROBOT_DATASET  # 指向您的数据集路径
        data_mix: bridge_rt_1
    ```
3) 使用Accelerate运行：
    ```bash
    base_vlm=Qwen/Qwen2.5-VL-3B-Instruct
    Framework_name=QwenGR00T
    run_root_dir=./results
    run_id=${Framework_name}

    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      starVLA/training/train_starvla.py \
      --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
      --framework.framework_py ${Framework_name} \
      --framework.qwenvl.base_vlm ${base_vlm} \
      --run_root_dir ${run_root_dir} \
      --run_id ${run_id} \
      --wandb_project your_project \
      --wandb_entity your_name
    ```

Accelerate原生支持多GPU；确切的启动命令取决于您的集群调度程序和设置（例如Slurm）。

```bash
srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
```

注意：`run_root_dir`存储统一的配置快照和数据处理元数据，以确保可重现性和快速重启。

</details>

<details id="train-smaller-vlm" close>
<summary><b>🚀 使用较小的VLM训练</b></summary>

```bash
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --machine_rank $SLURM_PROCID \
      --num_machines $SLURM_NNODES \
      --num_processes=${TOTAL_GPUS} \
      starVLA/training/train_starvla.py \
      --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
      --framework.framework_py QwenGR00T \
      --framework.qwenvl.base_vlm microsoft/Florence-2-large \
      --run_root_dir ${run_root_dir} \
      --run_id ${run_id} \
      --wandb_project your_project \
      --wandb_entity your_name
```

注意：为确保与已发布的检查点更好的兼容性，我们继续使用`--framework.qwenvl`。此参数将在下一个版本中统一。

</details>

## 📖 常见问题解答

<details close>
<summary><b>问：为什么不在数据加载器中进行预处理？</b></summary>

答：我们对此进行了性能分析：数据预处理耗时不到1%。将其保留在框架内是可以接受的，并允许模型特定的灵活处理。

</details>

<details close>
<summary><b>问：我可以使用除Qwen2.5-VL之外的其他骨干网络吗？</b></summary>

答：可以。实现新的视觉+语言模块并在框架内组合它们；任何其他现有模型都可以交换使用。由于框架处理原始动作数据，因此很容易进行交换。

</details>

<details close>
<summary><b>问：为什么视觉塔没有抽象接口？</b></summary>

答：我们认为VLM将成为基础模型，并且本身会具有其原生的视觉塔。

</details>

<details close>
<summary><b>问：可以通过终端覆盖或添加参数吗？</b></summary>

答：可以。我们使用OmegaConf.load(args.config_yaml)作为单一配置入口；独立调试也使用args.config_yaml。参数可能是有意冗余的；您可以通过CLI自由添加或覆盖它们。

示例：
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml  \
  --num_processes 8 \
  starVLA/training/train_internvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct \ # 覆盖框架选择
  --framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct \ # 覆盖框架选择
  --framework.action_model.new_module ${module_name} \ # 向动作模型插入新模块
```

⚠️: `framework.action_model.new_module`只会添加到全局配置中；其行为取决于您的框架。

</details>

<details close>
<summary><b>问：可以通过参数冻结VLM吗？</b></summary>

答：可以。StarVLA使用正则表达式/名称列表控制冻结。示例：
```
--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder" \
```
提示：您可以先`print(your_model)`查看模块的相对路径，并将其列为逗号分隔值。
（实现在[TrainerUtils.freeze_backbones](file:///home/lzr/robot/starVLA/starVLA/training/trainer_utils/trainer_tools.py#L150-L192)中。）

</details>

<details close>
<summary><b>问：我可以为不同模块设置不同的学习率吗？</b></summary>

答：可以，starVLA也使用name:value字典控制学习组。配置示例：
```yaml
trainer:
  learning_rate:
    base: 1e-05      # 其他模块
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```
（也参考`trainer_tools.build_param_lr_groups`。）
</details>

<details close>
<summary><b>问：我可以从检查点恢复训练吗？</b></summary>

答：可以。在`config.yaml`中指定最新的检查点路径，例如：
```yaml
trainer:
  pretrained_checkpoint: path_to_steps_10000.pt
  reload_modules: "action_model"
```
空的`reload_modules`意味着完全加载所有模型。但是，starVLA不保存`optimizer state`。这需要大量内存/磁盘且带来的好处有限。

</details>

## ✍️ 引用与版权

StarVLA在MIT许可证下发布，允许商业使用、修改、分发和私人使用。允许派生作品进行变基；当从上游StarVLA变基时，请使用描述性提交消息（例如"chore: rebase from StarVLA"）并至少保留最近两次上游提交为单独提交。详见[许可证](LICENSE)。

```
@misc{starvla2025,
  title  = {StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  author = {starVLA Community},
  url = {https://github.com/starVLA/starVLA}
  year   = {2025}
}
```

## 🤝 贡献

1) 如果您发现问题，请先开启一个Issue。如果问题持续存在或需要澄清，请发起讨论，我们会跟进。

2) 如果您有改进StarVLA的想法，请随时开启PR。为确保我们会接受您的贡献，请先通过Issue对齐范围和设计，或通过这个[合作表单](https://forms.gle/R4VvgiVveULibTCCA)预约简短同步会议。

3) 如果您遇到阻碍或想要头脑风暴，请填写[合作表单](https://forms.gle/R4VvgiVveULibTCCA)。我们每周五下午举办办公时间进行现场讨论。

提示：提交PR之前，请在本地运行make check以通过格式化和lint检查。

## 🙏 致谢

该项目从多个著名的开源项目中汲取灵感和参考，包括：
- [LeRobot](https://github.com/huggingface/lerobot)  
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)  
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)  
- [Qwen-VL](https://github.com/QwenLM/Qwen3-VL/tree/main)  
- [InternVL](https://github.com/OpenGVLab/InternVL)  

代码库最初是从[InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1)派生的。