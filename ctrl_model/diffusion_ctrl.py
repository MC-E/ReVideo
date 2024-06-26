from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Union

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from sgm.models.diffusion import DiffusionEngine
from sgm.util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)
from sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
import cv2
import numpy as np
from utils.save_video import save_rgb_video, save_flow_video

class DiffusionEngineCtrl(DiffusionEngine):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        controlnet_config: Optional[Dict] = None,
        ctrlnet_key: Optional[str] = None,
        items = None,
        probabilities = None,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        kernel_size = 199,
        sigma = 20,
    ):
        super().__init__(
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config,
            sampler_config,
            optimizer_config,
            scheduler_config,
            loss_fn_config,
            network_wrapper,
            ckpt_path,
            use_ema,
            ema_decay_rate,
            scale_factor,
            disable_first_stage_autocast,
            input_key,
            log_keys,
            no_cond_log,
            compile_model,
            en_and_decode_n_samples_a_time,
        )
        self.items = items
        self.probabilities = probabilities
        ctrlnet_model = instantiate_from_config(controlnet_config)
        if ctrlnet_model.ctrlnet_ckpt_path is not None:
            ctrlnet_model.init_from_ckpt()
        elif ctrlnet_model.if_init_from_unet == True:
            ctrlnet_model.init_from_unet(self.model.diffusion_model)
        else:
            print('random initial UNet')
        self.ctrlnet_key = ctrlnet_key
        self.model = CtrlNetWrapper(
            self.model.diffusion_model,
            compile_model=False,  # the UNet may be compiled in the super().__init__()
            ctrlnet_model=ctrlnet_model,
        )
        

class CtrlNetWrapper(OpenAIWrapper):
    def __init__(self, diffusion_model, compile_model: bool = False, ctrlnet_model: nn.Module = None):
        super().__init__(diffusion_model, compile_model)
        self.ctrlnet_model = ctrlnet_model

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        ctrlnet_cond = c.get("ctrl_input")
        mask = c.get("mask")
        region = c.get("region")
        assert ctrlnet_cond is not None, "Input SVD CtrlNet conditon is None!!!"
        down_block_res_samples, mid_block_res_sample = self.ctrlnet_model(
            x,
            mask=mask,
            region=region,
            conds=ctrlnet_cond,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            **kwargs,
        )