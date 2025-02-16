# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import json
from ..utils import PeftConfig, PeftType, transpose


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    blc_alpha: int = field(default=None, metadata={"help": "Alpha of blcloss"})
    blc_weight: int = field(default=None, metadata={"help": "Weight of blcloss"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    setting: str = field(default=None, metadata={"help": "Setting of Lora"})


    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model): # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias, config.setting)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode) and not is_hf_device_map_available,
            "setting": self.peft_config.setting,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", setting = None) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        if setting == "stage3.1":
            if "expert_route_user_embedding" in n or "user_embedding" in n or "expert_route_input" in n or "expert_route_lora" in n or "A0" in n or "A1" in n  or "A2" in n or "A3" in n or "A4" in n or "B0" in n or "B1" in n or "B2" in n or "B3" in n or "B4" in n:
                p.requires_grad = False
        elif setting == "stage2":
            if "lora_A_user" in n or "lora_B_user" in n or "expert_route_lora" in n:
                p.requires_grad = False
        elif setting == "stage3.2":
            if "lora_A_user" in n or "lora_B_user" in n or "expert_route_user_embedding" in n or "user_embedding" in n or "expert_route_input" in n or "A0" in n or "A1" in n  or "A2" in n or "A3" in n or "A4" in n or "B0" in n or "B1" in n or "B2" in n or "B3" in n or "B4" in n:
                p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        setting: str = "router_x_wo_user",
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        
        self.fan_in_fan_out = fan_in_fan_out
        self.setting = setting
        # Actual trainable parameters
        if r > 0:
            self.lora_expert_route_input = nn.Linear(in_features, self.lora_num, bias=False)
            # expert route, input is x 
            self.lora_expert_route_lora = nn.Linear(out_features, self.lora_num, bias=False)
            self.lora_expert_route_user_embedding = nn.Linear(in_features, self.lora_num, bias=False)
            # expert route, input is user embedding
            self.lora_user_embedding = nn.Embedding(101, in_features)
            # get user embedding
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            # 5 lora
            for i in range(1, 102):
                setattr(self, f"lora_A_user{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B_user{i}", nn.Linear(r, out_features, bias=False))
            # 100 lora
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)
            for i in range(1,102):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A_user{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B_user{i}").weight)

            nn.init.kaiming_uniform_(self.lora_expert_route_input.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_expert_route_lora.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_expert_route_user_embedding.weight, a=math.sqrt(5))
            


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_expert_route_input.train(mode)
        self.lora_expert_route_user_embedding.train(mode)
        self.lora_expert_route_lora.train(mode)
        self.lora_user_embedding.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)
        for i in range(1, 102):
            getattr(self, f"lora_A_user{i}").train(mode)
            getattr(self, f"lora_B_user{i}").train(mode)        

    def eval(self):
        nn.Linear.eval(self)
        self.lora_expert_route_input.eval()
        self.lora_expert_route_user_embedding.eval()
        self.lora_expert_route_lora.eval()
        self.lora_user_embedding.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()
        for i in range(1, 102):
            getattr(self, f"lora_A_user{i}").eval()
            getattr(self, f"lora_B_user{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, user_id=None):

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(") 
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            M_omega = []
            if self.r > 0:
                if self.setting == "stage2":
                    route_weight_input = nn.functional.softmax(self.lora_expert_route_input(x), dim=-1, dtype=torch.float32).to(result.dtype)
                    user_id_device = result.device
                    user_id_value = user_id.to(user_id_device).item() - 1
                    task_tensor = torch.tensor([user_id_value], device=user_id_device)
                    embedding_result = self.lora_user_embedding(task_tensor).to(result.device)
                    route_weight_user = nn.functional.softmax(self.lora_expert_route_user_embedding(embedding_result), dim=-1, dtype=torch.float32).to(result.dtype)
                    route_mix = nn.functional.softmax(route_weight_input + route_weight_user, dim=-1, dtype=torch.float32).to(result.dtype)
                    for i in range(self.lora_num):
                        result = result + 0.2 * torch.unsqueeze(route_mix[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                        M_omega.append(torch.unsqueeze(route_mix[:,:,i], -1))
                elif self.setting == "stage3.1":
                    route_weight_input = nn.functional.softmax(self.lora_expert_route_input(x), dim=-1, dtype=torch.float32).to(result.dtype)
                    user_id_device = result.device
                    user_id_value = user_id.to(user_id_device).item() - 1
                    task_tensor = torch.tensor([user_id_value], device=user_id_device)
                    embedding_result = self.lora_user_embedding(task_tensor).to(result.device)
                    route_weight_user = nn.functional.softmax(self.lora_expert_route_user_embedding(embedding_result), dim=-1, dtype=torch.float32).to(result.dtype)
                    route_mix = nn.functional.softmax(route_weight_input + route_weight_user, dim=-1, dtype=torch.float32).to(result.dtype)
                    for i in range(self.lora_num):
                        result = result + 1/self.lora_nums * torch.unsqueeze(route_mix[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                    result = result + getattr(self, f"lora_B_user{user_id.item()}")(getattr(self, f"lora_A_user{user_id.item()}")(self.lora_dropout(x))) * self.scaling
                elif self.setting == "stage3.2":
                    lora_out = getattr(self, f"lora_B_user{user_id.item()}")(getattr(self, f"lora_A_user{user_id.item()}")(self.lora_dropout(x))) * self.scaling
                    route_weight_lora = nn.functional.softmax(self.lora_expert_route_lora(lora_out), dim=-1, dtype=torch.float32).to(result.dtype)
                    for i in range(self.lora_num):
                        result = result + 0.2 * torch.unsqueeze(route_weight_lora[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                        M_omega.append(torch.unsqueeze(route_weight_lora[:,:,i], -1))
                    result = result + lora_out
                else:
                    print("Error: setting not found")
        blcls = torch.zeros(1)[0].to(result)
        if self.setting == "stage3.1":
            blcls = 0
        else:
            if user_id != None:
                averages = [torch.mean(tensor).item() for tensor in M_omega]
                M_omega = torch.tensor(averages)
                M_s = torch.matmul(M_omega.view(-1,1), M_omega.view(1,-1))
                L_s = torch.abs(M_s)
                L_s = L_s - torch.eye(L_s.size(0))*L_s  
                blcls = torch.sum(L_s)/(32*3)
        return result, blcls