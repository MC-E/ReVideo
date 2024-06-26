from functools import partial

from typing import List, Optional, Union, Tuple

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

from sgm.modules.diffusionmodules.openaimodel import *
from sgm.modules.video_attention import SpatialVideoTransformer
from sgm.util import default
from sgm.modules.diffusionmodules.util import AlphaBlender
from sgm.modules.diffusionmodules.video_model import VideoResBlock, VideoUNet
import numpy as np
import cv2

class ControledVideoUnet(VideoUNet):
    """
    Only modify the forward function by adding additional controls
    """
    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        ## tbd: check the role of "image_only_indicator"
        num_video_frames = self.num_frames
        image_only_indicator = torch.zeros(
                    x.shape[0]//num_video_frames, num_video_frames
                ).to(x.device) if image_only_indicator is None else image_only_indicator

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        ## x shape: [bt,c,h,w]
        h = x
        hs = []
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        # svd ctrl
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual
        for module in self.output_blocks:
            if down_block_additional_residuals is not None:
                h = th.cat([h, hs.pop() + down_block_additional_residuals.pop()], dim=1)
            else:
                h = th.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )

        return self.out(h)


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class MaskEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class WeightEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels*2, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )
        self.sig = nn.Sigmoid()
        self.conv_final = nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning, t, cond_embeddings, mask_embeddings):
        t_cond = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(conditioning.shape)
        conditioning = torch.cat([conditioning, t_cond], dim=1)
        w = self.conv_in(conditioning)
        w = F.silu(w)

        for block in self.blocks:
            w = block(w)
            w = F.silu(w)

        w = self.conv_out(w)
        w = self.sig(w)

        embedding = cond_embeddings*w+mask_embeddings*(1-w)
        embedding = self.conv_final(embedding)

        return embedding


class VideoCtrlNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_frames: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        # ctrlnet
        conditioning_channels: int = 3,
        conditioning_channels_mask: int = 1,
        conditioning_channels_region: int = 1,
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        ctrlnet_scale: float = 1.0,
        ctrlnet_ckpt_path: Optional[str] = None,
        if_init_from_unet = True,
    ):
        super(VideoCtrlNet, self).__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_frames = num_frames
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )
        self.dims = dims
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.if_init_from_unet = if_init_from_unet
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.ctrlnet_scale = ctrlnet_scale
        self.ctrlnet_ckpt_path = ctrlnet_ckpt_path

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=model_channels,
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )
        self.mask_embedding = MaskEmbedding(
            conditioning_embedding_channels=model_channels,
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels_mask,
        )
        self.weight_embedding = WeightEmbedding(
            conditioning_embedding_channels=model_channels,
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels_region,
        )

        self.controlnet_down_blocks = nn.ModuleList([
            self.make_zero_conv(model_channels),
        ])

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                # ctrlnet blocks
                self.controlnet_down_blocks.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.controlnet_down_blocks.append(self.make_zero_conv(ch))
                self._feature_size += ch
                # self.step_cur = 0

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        # ctrlnet mid block
        self.controlnet_mid_block = self.make_zero_conv(ch)

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def init_from_ckpt(self, ckpt_path=None):
        path = self.ctrlnet_ckpt_path
        if ckpt_path is not None:
            path = ckpt_path

        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
            sd = filter(lambda x: x[0].startswith("model.ctrlnet_model"), sd.items())
            sd = {k.replace("model.ctrlnet_model.", ""): v for k, v in sd}
        elif path.endswith("bin"):
            sd = torch.load(path, map_location="cpu")
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"@CtrlNet: Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def init_from_unet(self, unet):
        self.input_blocks.load_state_dict(unet.input_blocks.state_dict())
        self.middle_block.load_state_dict(unet.middle_block.state_dict())
        print("init from unet successfully!")

    def forward(
        self,
        x: th.Tensor,
        conds: th.Tensor,
        mask: th.Tensor,
        region: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        ## tbd: check the role of "image_only_indicator"
        num_video_frames = self.num_frames
        image_only_indicator = torch.zeros(
                    x.shape[0]//num_video_frames, num_video_frames
                ).to(x.device) if image_only_indicator is None else image_only_indicator

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        # ctrlnet
        cond_embeddings = self.controlnet_cond_embedding(conds)
        mask = mask.permute(0,2,1,3,4)
        mask = mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2], mask.shape[3], mask.shape[4])
        mask_embeddings = self.mask_embedding(mask)
        cond_embeddings = self.weight_embedding(region, timesteps, cond_embeddings, mask_embeddings) ##*0.1

        ## x shape: [bt,c,h,w]
        h = x
        down_block_res_samples = []
        for module, zero_module in zip(self.input_blocks, self.controlnet_down_blocks):
            if cond_embeddings is not None:
                # for the conv_in block
                h = module(
                    h,
                    emb,
                    context=context,
                    image_only_indicator=image_only_indicator,
                    time_context=time_context,
                    num_video_frames=num_video_frames,
                )
                h += cond_embeddings
                cond_embeddings = None
            else:
                h = module(
                    h,
                    emb,
                    context=context,
                    image_only_indicator=image_only_indicator,
                    time_context=time_context,
                    num_video_frames=num_video_frames,
                )
            down_block_res_samples.append(
                zero_module(
                    h,
                    emb,
                    context=context,
                    image_only_indicator=image_only_indicator,
                    time_context=time_context,
                    num_video_frames=num_video_frames,
                ) * self.ctrlnet_scale
            )

        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        mid_block_res_sample = self.controlnet_mid_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        ) * self.ctrlnet_scale
        return down_block_res_samples, mid_block_res_sample


