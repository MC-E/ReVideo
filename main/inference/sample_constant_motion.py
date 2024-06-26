import datetime, time
import os, sys, argparse
import math
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from sgm.util import default, instantiate_from_config
from utils.save_video import save_flow_video, save_rgb_video
import torch.nn as nn
from utils.visualizer import Visualizer
import time
from utils.tools import resize_pil_image, quick_freeze, get_gaussian_kernel, get_batch, get_unique_embedder_keys_from_conditioner, load_model

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
if not os.path.exists('ckpt/model.ckpt'):
    torch.hub.download_url_to_file(
        'https://huggingface.co/Adapter/ReVideo/resolve/main/model.ckpt',
        'ckpt/model.ckpt')

def sample(
    input_path: str = "outputs/inputs/test_image.png",  # Can either be image file or folder with image files
    path_ref: str = None,
    ckpt: str = "checkpoints/svd.safetensors",
    config: str = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    save_fps: int = 10,
    resize: Optional[bool] = False,
    # points = None
    s_w = None, 
    e_w = None, 
    s_h = None, 
    e_h = None,
    ps_h = None,
    ps_w = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    # flow
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
    guassian_filter = quick_freeze(get_gaussian_kernel(kernel_size=51, sigma=10, channels=2)).to(device)
    visualizer = Visualizer(tracks_leave_trace=-1, show_first_frame=0, mode='cool', linewidth=2)
    visualizer_layer = Visualizer(tracks_leave_trace=-1, show_first_frame=0, mode='cool', linewidth=4)

    torch.manual_seed(seed)

    all_img_paths = os.listdir(input_path)
    all_img_paths.sort()
    for i in range(len(all_img_paths)):
        all_img_paths[i] = os.path.join(input_path, all_img_paths[i])

    print(f'loaded {len(all_img_paths)} images.')
    os.makedirs(output_folder, exist_ok=True)
    images = []
    for no, input_img_path in enumerate(all_img_paths):
        filepath, fullflname = os.path.split(input_img_path)
        filename, ext = os.path.splitext(fullflname)
        print(f'-sample {no+1}: {filename} ...')
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            w_org, h_org = image.size
            image = resize_pil_image(image, max_resolution=1024*1024)
            # image = resize_pil_image(image, max_resolution=896*896)
            scale_h = image.size[0]/w_org
            scale_w = image.size[1]/h_org
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        images.append(image)
    images = images[:num_frames]#[::-1]#.flip(dims=(1,))
    images = torch.stack(images, dim=2)
    save_rgb_video((images.flip(1)+1)/2.,'outputs/org.mp4')

    img_ref = Image.open(path_ref)
    if img_ref.mode == "RGBA":
        img_ref = img_ref.convert("RGB")
    img_ref = img_ref.resize((images.shape[-1], images.shape[-2]))
    img_ref = ToTensor()(img_ref)
    img_ref = img_ref * 2.0 - 1.0
    img_ref = img_ref.unsqueeze(0).to(device)

    with torch.no_grad():
        vid_cotracker = ((images+1)/2.).permute(0,2,1,3,4)*255.
        grid = torch.zeros(1,len(ps_h),3)
        for i in range(len(ps_h)):
            grid[0,i,1] = ps_h[i]*scale_h
            grid[0,i,2] = ps_w[i]*scale_w
        grid = grid.to(device)
        tracks, _ = cotracker(vid_cotracker, queries=grid) # B T N 2,  B T N 1
    
    layer = torch.ones_like(images.permute(0,2,1,3,4))
    res_video = visualizer_layer.visualize(layer.cpu()*255., tracks=tracks, save_video=False)
    res_video = (
        (rearrange(res_video[0], "t c h w -> t h w c"))
        .numpy()
        .astype(np.uint8)
    )
    frame = cv2.cvtColor(res_video[-1], cv2.COLOR_RGB2BGR)
    filename = input_path.split('/')[-1]
    cv2.imwrite(os.path.join('vis_im', filename, 'layer.png'), frame)

    b,c,n,w,h = images.shape
    s_w = int(s_w*scale_w)
    e_w = int(e_w*scale_w)
    s_h = int(s_h*scale_h)
    e_h = int(e_h*scale_h)
    model_config = config

    model, filter = load_model(
        model_config,
        ckpt,
        device,
        num_frames,
        num_steps,
    )


    select_point = tracks.flip(-1)
    maps = []
    for i in range(num_frames-1):
        map = torch.zeros((1,2,img_ref.shape[-2],img_ref.shape[-1])).to(device)
        rows = select_point[:, i+1, :, 0].to(device).int()#, dtype=torch.int64)
        cols = select_point[:, i+1, :, 1].to(device).int()#, dtype=torch.int64)
        rows = torch.clip(rows, 0, w-1)
        cols = torch.clip(cols, 0, h-1)
        map[0,:,rows[0], cols[0]] = (select_point[0, i+1] - select_point[0, i]).permute(1,0)#flow[kk, jj+1, :, rows[kk], cols[kk]].clone()#.cpu()
        maps.append(map)

    maps = [torch.zeros_like(maps[0])]+maps
    maps = torch.stack(maps, dim=2)#.reshape(b*n,2,w,h)
    guassian_filter = quick_freeze(get_gaussian_kernel(kernel_size=51, sigma=10, channels=2)).to(device)
    with torch.no_grad():
        maps = maps.permute(0,2,1,3,4).reshape(b*n,2,w,h)
        maps = guassian_filter(maps).reshape(b,n,2,w,h)
    images[:,:,:,s_w:e_w,s_h:e_h] = -1

    save_flow_video(maps.permute(0,2,1,3,4), 'outputs/flow.mp4')
    save_rgb_video((images.flip(1)+1)/2.,'outputs/content.mp4')

    flow = maps.reshape(b*n,2,w,h)

    region = torch.zeros_like(images)[:,:1]
    region[:,:,:,s_w:e_w,s_h:e_h] = 1
    region = region.permute(0,2,1,3,4).reshape(b*n,1,w,h)

    value_dict = {}
    print(images.shape)
    value_dict["video"] = images.to(dtype=torch.float16)
    value_dict["region"] = region.to(dtype=torch.float16)
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = img_ref.to(dtype=torch.float16) #images[:,:,0]
    value_dict["cond_frames"] = (img_ref + cond_aug * torch.randn_like(images[:,:,0])).to(dtype=torch.float16)
    print(cond_aug)

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device, dtype = torch.float16)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device, dtype=torch.float16)
            #additional_model_inputs["image_only_indicator"][:,0] = 1
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            c['mask'] = images.clone()
            uc['mask'] = images.clone()
            c['region'] = region.clone()
            uc['region'] = region.clone()
            c['ctrl_input'] = flow.clone()
            uc['ctrl_input'] = flow.clone()

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            filename = input_path.split('/')[-1]
            res_video = visualizer.visualize(samples.unsqueeze(0).cpu()*255., tracks=tracks, save_video=False)
            visualizer.save_video(res_video, filename='result_cotracker_%s'%filename, writer=None, step=0)
            os.makedirs(os.path.join('vis_im', filename, 'im_w_track'), exist_ok=True)
            os.makedirs(os.path.join('vis_im', filename, 'im_wo_track'), exist_ok=True)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            vid_wo_track = (
                (rearrange(res_video[0], "t c h w -> t h w c"))
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            for idx, frame in enumerate(vid):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                name = '%04d.png'%idx
                cv2.imwrite(os.path.join('vis_im', filename, 'im_wo_track', name), frame)
            for idx, frame in enumerate(vid_wo_track):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                name = '%04d.png'%idx
                cv2.imwrite(os.path.join('vis_im', filename, 'im_w_track', name), frame)
    
    print(f'Done! results saved in {output_folder}.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--input", type=str, default=None, help="image path or folder")
    parser.add_argument("--path_ref", type=str, default=None, help="reference image path")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=int, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=6, help="control the fps")
    parser.add_argument("--motion", type=int, default=127, help="control the motion magnitude")
    parser.add_argument("--cond_aug", type=float, default=0.02, help="adding noise to input image")
    parser.add_argument("--decoding_t", type=int, default=1, help="frames num to decoding per time")
    parser.add_argument("--resize", action='store_true', default=False, help="resize all input to default resolution")
    parser.add_argument("--s_w", type=int, default=None)
    parser.add_argument("--e_w", type=int, default=None)
    parser.add_argument("--s_h", type=int, default=None)
    parser.add_argument("--e_h", type=int, default=None)
    parser.add_argument("--ps_h", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--ps_w", metavar="N", type=int, nargs="+", default=None)
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@SVD Inference: %s"%now)
    #Fire(sample)
    parser = get_parser()
    args = parser.parse_args()
    sample(input_path=args.input, path_ref=args.path_ref, ckpt=args.ckpt, config=args.config, num_frames=args.frames, num_steps=args.ddim_steps, \
        fps_id=args.fps, motion_bucket_id=args.motion, cond_aug=args.cond_aug, seed=args.seed, \
        decoding_t=args.decoding_t, output_folder=args.savedir, save_fps=args.savefps, resize=args.resize, s_w=args.s_w, e_w=args.e_w, s_h=args.s_h, e_h=args.e_h, ps_w=args.ps_w, ps_h=args.ps_h)
