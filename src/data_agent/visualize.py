import os
import shutil
import sys
from pathlib import Path

import einops
import numpy as np
import argparse

from PIL import Image
from carla_birdeye_view import BirdViewProducer, BirdViewMasks
from einops import rearrange

from data_agent.data_io import load_data
from data_agent.render import map_seg_image, draw_bboxes_on_image


def birdview_compress(birdview):
    index = birdview[::-1].argmax(axis=0).astype(np.uint8)
    return np.where(birdview.sum(axis=0), index + 1, np.zeros_like(index))


def birdview_decompress(birdview):
    H, W = birdview.shape
    n_channels = len(BirdViewMasks)
    birdview = np.eye(n_channels + 1, dtype=np.uint8)[birdview.reshape(-1)]
    birdview = rearrange(birdview, "(h w) c -> c h w", h=H)
    mask, birdview = birdview[0], birdview[1:]
    return birdview[::-1] * (1 - mask)


def gen_debug_images(tick_data):
    debug_images = {}

    pos_keys = tick_data["bbox"].keys()

    rgb_imgs = rearrange(tick_data["rgb"].copy(), "h (p w) c -> p h w c", p=len(pos_keys))
    debug_list = []
    for rgb_img, (pos, bbox) in zip(rgb_imgs, tick_data["bbox"].items()):
        debug_img = draw_bboxes_on_image(rgb_img, bbox, tick_data["agent_info"]["control"].get("main", {}).get("hazards", {}))
        debug_list.append(debug_img)
        debug_images[f"rgb_bbox_{pos}"] = debug_img

    debug_img = np.concatenate(debug_list, axis=1)

    if "rgb_high_res" in tick_data:
        rgb_high_res = tick_data["rgb_high_res"]
        H, W, C = rgb_high_res.shape
        target_W = debug_img.shape[1]
        if target_W % W == 0:
            scale = target_W // W
            rgb_high_res = np.repeat(np.repeat(rgb_high_res, scale, axis=0), scale, axis=1)
        else:
            target_H = int(target_W * H / W) // 2 * 2  # make sure it is devisible by 2
            rgb_high_res = np.array(Image.fromarray(rgb_high_res).resize((target_W, target_H)))
        debug_img = np.concatenate([rgb_high_res, debug_img], axis=0)

    if "seg" in tick_data:
        seg_img = map_seg_image(tick_data["seg"])
        debug_img = np.concatenate([debug_img, seg_img], axis=0)

        seg_imgs = rearrange(seg_img, "h (p w) c -> p h w c", p=len(pos_keys))
        for pos, seg_img in zip(pos_keys, seg_imgs):
            debug_images[f"seg_{pos}"] = seg_img

    size = debug_img.shape[0] // 2 if "seg" in tick_data else debug_img.shape[0]
    topdown = map_seg_image(tick_data["topdown"])
    debug_images["topdown"] = topdown
    topdown = np.array(Image.fromarray(topdown).resize((size, size)))

    birdview = tick_data["birdview"]
    birdview = BirdViewProducer.as_rgb(birdview_decompress(birdview))
    debug_images["birdview"] = birdview
    birdview = np.array(Image.fromarray(birdview).resize((size, size)))

    topdown_img = np.concatenate([birdview, topdown], axis=0 if "seg" in tick_data else 1)

    if "lidar_topdown" in tick_data:
        lidar_img = (tick_data["lidar_topdown"] * 255).astype(np.uint8)
        lidar_img = einops.repeat(lidar_img, "h w -> h w c", c=3)
        lidar_img = np.array(Image.fromarray(lidar_img).resize((size, 2 * size) if "seg" in tick_data else (size // 2, size)))

        debug_images["main"] = np.concatenate([debug_img, lidar_img, topdown_img], axis=1)
    else:
        debug_images["main"] = np.concatenate([debug_img, topdown_img], axis=1)

    return debug_images


def create_debug_images(data_dir: str):
    data_dir = Path(data_dir)
    debug_img_dir = data_dir / "debug_extra"
    if os.path.exists(debug_img_dir):
        shutil.rmtree(debug_img_dir)

    for i in range(len(os.listdir(data_dir / "rgb"))):
        sys.stdout.write(f"Processing index {i}\r")
        sys.stdout.flush()

        data = load_data(data_dir, i, as_carla_objects=True)
        debug_images = gen_debug_images(data)

        if i == 0:
            for img_type in debug_images:
                (debug_img_dir / img_type).mkdir(parents=True, exist_ok=True)
        for img_type, img in debug_images.items():
            Image.fromarray(img).save(
                debug_img_dir / img_type / ("%04d.jpg" % i)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    create_debug_images(args.data)
