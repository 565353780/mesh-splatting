import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm



# python extract_normals.py -s /gpfs/scratch/acad/telim/datasets/tandt/truck/

def resize_to_multiple(tensor, multiple=28):
    B, C, H, W = tensor.shape
    new_H = (H // multiple) * multiple
    new_W = (W // multiple) * multiple
    return torch.nn.functional.interpolate(
        tensor, size=(new_H, new_W), mode='bilinear', align_corners=False
    )

def main():
    parser = argparse.ArgumentParser(
        description="Compute normal maps using Metric3D for all jpg images in a folder."
    )
    parser.add_argument(
        "-s", "--scene",
        required=True,
        help="Path to the scene root, for example /gpfs/scratch/.../MipNeRF360/bicycle"
    )
    parser.add_argument(
        "-i", "--images_folder",
        default="images",
        help="Images subfolder name, default is 'images'. "
             "Use 'images_2' or 'images_4' to override."
    )
    args = parser.parse_args()

    scene_path = args.scene
    images_folder = args.images_folder

    # Input folder: scene_root / images_folder
    input_dir = os.path.join(scene_path, images_folder)

    # Output folder: map images -> normals, images_2 -> normals_2, etc
    if images_folder.startswith("images"):
        output_folder = images_folder.replace("images", "normals", 1)
    else:
        output_folder = "normals"

    output_dir = os.path.join(scene_path, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scene path      : {scene_path}")
    print(f"Images folder   : {input_dir}")
    print(f"Normals folder  : {output_dir}")

    # Load model (FP16)
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.eval().cuda().half()

    transform = transforms.ToTensor()

    # Collect all JPG images
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
    ])

    if not images:
        print("No jpg images found in", input_dir)
        return

    # Process with progress bar
    for fname in tqdm(images, desc="Processing images"):
        img_path = os.path.join(input_dir, fname)

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        original_size = (orig_h, orig_w)

        rgb = transform(image).unsqueeze(0).half().cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            rgb_resized = resize_to_multiple(rgb, multiple=28)
            pred_depth, _, output_dict = model.inference({"input": rgb_resized})
            pred_normal = output_dict["prediction_normal"][:, :3]

            pred_normal_resized = torch.nn.functional.interpolate(
                pred_normal, size=original_size, mode="bilinear", align_corners=False
            )

        normals = pred_normal_resized.squeeze().float().cpu().numpy()
        normals = np.transpose(normals, (1, 2, 0))
        normals_vis = ((normals + 1.0) / 2.0).clip(0, 1)
        normals_uint8 = (normals_vis * 255).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR))

        del rgb, rgb_resized, pred_depth, output_dict, pred_normal, pred_normal_resized
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
