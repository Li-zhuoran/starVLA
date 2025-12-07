# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)

Qwen-OFT 框架

一种轻量级实现，通过在视觉语言模型（VLM）中引入一个动作特殊标记（action special token），并基于多视角图像和语言指令（与 VLM 共享参数）并行预测连续动作。  
该方法受 OpenVLA-Oft 启发。

关键要点：
- 基于 Qwen2.5 视觉语言主干网络（vision-language backbone）  
- 在 VLM 中注入一个动作特殊标记（action special token）  
- 通过对该动作特殊标记的隐藏状态进行 L1 回归，实现连续动作的预测  

注：如何为 Qwen2.5 添加特殊标记：
- 下载我们已添加特殊标记的模型权重：https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action  
- 或参考 /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md（需稍作代码适配）
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images

@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    多模态视觉-语言-动作模型，使用Qwen2.5-VL作为骨干网络。

    组件:
      - Qwen2.5 VL接口用于融合的语言/视觉token嵌入
      - 层级QFormer用于多层次特征聚合
      - DINO编码器用于密集多视角空间tokens
      - DiT扩散头用于未来动作序列建模

    重点: 基于图像和指令条件预测未来的连续动作。

    Attributes:
        config: 模型配置信息
        qwen_vl_interface: Qwen视觉语言接口模型
        action_model: 动作预测模型
        future_action_window_size: 未来动作窗口大小
        past_action_window_size: 过去动作窗口大小
        chunk_len: 动作块长度
        hidden_dim: 隐藏层维度
        action_token: 动作特殊标记
        action_token_id: 动作特殊标记ID
        l1_loss: L1损失函数
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        构造所有子模块并缓存关键配置值。

        Args:
            config: 包含框架和训练器部分的分层配置(OmegaConf/dict)。
            **kwargs: 保留供将来覆盖使用(未使用)。
        """
        super().__init__()
        # 初始化模型配置
        self.config = config
        # 获取Qwen视觉语言模型接口
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # 对齐维度，将视觉语言模型的隐藏层大小设置到动作模型配置中
        # align dims --> we should put them to config or no?
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        # 初始化动作预测模型
        self.action_model = get_action_model(config=self.config)

        # 设置动作窗口大小参数
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        # 计算动作块总长度：过去动作窗口 + 当前动作 + 未来动作窗口
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        # 设置隐藏层维度
        self.hidden_dim = config.framework.action_model.action_hidden_dim
        
        # 定义动作特殊标记及其ID
        self.action_token = "🔍" # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # 初始化L1损失函数用于动作预测
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向传播：直接回归未来动作（无扩散）。

        执行流程:
          1. 构建QwenVL输入（图像+指令tokens）
          2. 从配置的层范围提取隐藏状态
          7. 预测动作并计算L1损失

        Args:
            examples: 样本列表，每个字典包含:
                - image: PIL.Image列表（多视角）
                - lang: 字符串形式的指令
                - action: 形状为[T, action_dim]的np.ndarray或列表
            **kwargs: 保留参数。

        Returns:
            dict:
                action_loss (torch.Tensor): 标量扩散噪声预测损失。
        """
        # 从样本中提取图像、指令和动作标签
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # step 0: add special action token to instruction
        # 在指令末尾添加动作预测提示和特殊标记
        action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        # 构建QwenVL模型输入格式
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            # 计算 L1 损失
            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        步骤:
          1. 将图像调整到训练分辨率（如果指定）
          2. 使用QwenVL进行编码（保留隐藏状态）
          6. 返回归一化的动作轨迹

        Args:
            batch_images: 样本列表；每个样本是PIL.Image列表（多视角）。
            instructions: 自然语言任务指令列表。
            cfg_scale: >1启用分类器自由指导（缩放条件vs无条件）。
            use_ddim: 是否使用DDIM确定性采样。
            num_ddim_steps: 如果启用DDIM的步数。
            **kwargs: 保留参数。

        Returns:
            dict:
                normalized_actions (np.ndarray): 形状为[B, T, action_dim]的扩散采样归一化动作。
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # step 0: add special action token to instruction
        action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,        # 可为 int 或 List[int]
    ) -> torch.Tensor:
        """
        向量化批量提取动作token嵌入:
          - 不再逐样本for循环
          - 取每个样本里最靠后的chunk_len个动作占位token
          
        Args:
            last_hidden: 最后一层隐藏状态，形状为[B, L, H]
            input_ids: 输入token IDs，形状为[B, L]
            action_token_id: 动作token ID，可以是int或List[int]

        Returns:
            action_queries: 动作查询嵌入，形状为[B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # 创建动作token的掩码，支持单个ID或多个ID列表
        # 支持多 id（如多个变体）
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin 需要 PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        # 检查每个样本中动作token的数量是否满足要求
        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        # 生成位置索引，并标记动作token的位置
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))   # 非动作位置置 -1

        # 提取每个样本中最靠后的chunk_len个动作token位置
        # 取最后 chunk_len 个（索引大的在序列靠后）
        # 注意: 已确保数量足够，不会出现 -1 被错误选中的问题
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values     # [B, chunk_len] 未排序
        # 按时间顺序排序位置索引
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # 根据选定的位置收集对应的隐藏状态向量
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_hidden_dim = 2048

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    # try get model
    model = Qwenvl_OFT(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")


    # # try forward model
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # zhe
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)
    # pass
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])