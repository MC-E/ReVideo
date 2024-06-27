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
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from utils.save_video import save_flow_video, save_rgb_video
import torch.nn as nn
from utils.visualizer import Visualizer
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
    ps_w = None, 
    pe_w = None, 
    ps_h = None, 
    pe_h = None,
    x_bias_all = None,
    y_bias_all = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    # flow
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
    images = images[:num_frames]
    images = torch.stack(images, dim=2)
    save_rgb_video((images.flip(1)+1)/2.,'outputs/org.mp4')

    img_ref = Image.open(path_ref)
    if img_ref.mode == "RGBA":
        img_ref = img_ref.convert("RGB")
    img_ref = img_ref.resize((images.shape[-1], images.shape[-2]))
    img_ref = ToTensor()(img_ref)
    img_ref = img_ref * 2.0 - 1.0
    img_ref = img_ref.unsqueeze(0).to(device)

    b,c,n,w,h = images.shape
    map = torch.zeros((1,2,num_frames-1,img_ref.shape[-2],img_ref.shape[-1])).to(device)
    region = torch.zeros_like(images)[:,:1]
    tracks = []
    assert len(x_bias_all)==len(y_bias_all) and len(x_bias_all)==len(s_w), 'Wrong bias!'
    for k in range(len(s_w)):
        s_w[k] = int(s_w[k]*scale_w)
        e_w[k] = int(e_w[k]*scale_w)
        s_h[k] = int(s_h[k]*scale_h)
        e_h[k] = int(e_h[k]*scale_h)
        images[:,:,:,s_w[k]:e_w[k],s_h[k]:e_h[k]] = -1
        region[:,:,:,s_w[k]:e_w[k],s_h[k]:e_h[k]] = 1

        p_start = [ps_h[k]*scale_h, ps_w[k]*scale_w]
        p_end = [pe_h[k]*scale_h, pe_w[k]*scale_w]

        x_dist = p_end[0]-p_start[0]
        y_dist = p_end[1]-p_start[1]
        inter_x = x_dist/(num_frames-2)
        inter_y = y_dist/(num_frames-2)
        x_bias = x_bias_all[k]
        y_bias = y_bias_all[k]
        for i in range(num_frames-1):
            x_cur = int(p_start[0]+i*inter_x + x_bias[i%len(x_bias)])
            y_cur = int(p_start[1]+i*inter_y + y_bias[i%len(y_bias)])
            if i == 0:
                x_per = p_start[0]
                y_per = p_start[1]
            else:
                x_per = int(p_start[0]+(i-1)*inter_x + x_bias[(i-1)%len(x_bias)])
                y_per = int(p_start[1]+(i-1)*inter_y+ y_bias[(i-1)%len(y_bias)])
            map[:,1,i,y_cur,x_cur] = x_cur - x_per
            map[:,0,i,y_cur,x_cur] = y_cur - y_per

        track = torch.zeros(b,num_frames,1,2)
        for i in range(num_frames):
            if i == 0:
                x_cur = p_start[0]
                y_cur = p_start[1]
            else:
                x_cur = int(p_start[0]+i*inter_x + x_bias[i%len(x_bias)])
                y_cur = int(p_start[1]+i*inter_y + y_bias[i%len(y_bias)])
            track[0,i,0,0]=x_cur
            track[0,i,0,1]=y_cur
        tracks.append(track)
    tracks = torch.cat(tracks, dim=2)

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

    pad = torch.zeros_like(map[:,:,:1,:,:])
    map = torch.cat([pad, map], dim=2)
    with torch.no_grad():
        map = map.permute(0,2,1,3,4).reshape(b*n,2,w,h)
        map = guassian_filter(map).reshape(b,n,2,w,h)

    save_flow_video(map.permute(0,2,1,3,4), 'outputs/flow.mp4')
    save_rgb_video((images.flip(1)+1)/2.,'outputs/content.mp4')

    flow = map.reshape(b*n,2,w,h)

    model_config = config

    model, filter = load_model(
        model_config,
        ckpt,
        device,
        num_frames,
        num_steps,
    )
    
    region = region.permute(0,2,1,3,4).reshape(b*n,1,w,h)

    value_dict = {}
    value_dict["video"] = images.to(dtype=torch.float16)
    value_dict["region"] = region.to(dtype=torch.float16)
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = img_ref.to(dtype=torch.float16) #images[:,:,0]
    value_dict["cond_frames"] = img_ref.to(dtype=torch.float16) #images[:,:,0] # + cond_aug * torch.randn_like(images[:,:,0])

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

            randn = torch.randn(shape, device=device, dtype=torch.float16)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device, dtype=torch.float16)
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
    parser.add_argument("--s_w", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--e_w", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--s_h", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--e_h", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--ps_w", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--pe_w", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--ps_h", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--pe_h", metavar="N", type=int, nargs="+", default=None)
    parser.add_argument("--x_bias_all", nargs="+", action="append", type=int, help="Horizontal swing")
    parser.add_argument("--y_bias_all", nargs="+", action="append", type=int, help="Vertical swing")
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@SVD Inference: %s"%now)
    #Fire(sample)
    parser = get_parser()
    args = parser.parse_args()
    sample(input_path=args.input, path_ref=args.path_ref, ckpt=args.ckpt, config=args.config, num_frames=args.frames, num_steps=args.ddim_steps, \
        fps_id=args.fps, motion_bucket_id=args.motion, cond_aug=args.cond_aug, seed=args.seed, \
        decoding_t=args.decoding_t, output_folder=args.savedir, save_fps=args.savefps, resize=args.resize, \
        s_w=args.s_w, e_w=args.e_w, s_h=args.s_h, e_h=args.e_h, ps_w=args.ps_w, pe_w=args.pe_w, ps_h=args.ps_h, \
        pe_h=args.pe_h, x_bias_all=args.x_bias_all, y_bias_all=args.y_bias_all)
