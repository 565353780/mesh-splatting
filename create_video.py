#
# Copyright (C) 2024, Inria, University of Liege, KAUST and University of Oxford
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  jan.held@uliege.be
#


import torch
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, TriangleModel
import numpy as np
from utils.render_utils import generate_path, create_videos
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- Helper for progressive zoom trajectory ---
def generate_zoom_trajectory(viewpoint_cameras, n_frames=480, zoom_start=0, zoom_duration=120, zoom_intensity=2.0):
    """
    Generate a camera trajectory with a progressive zoom in and out.
    zoom_start: frame index to start zoom in
    zoom_duration: number of frames for zoom in (zoom out will be symmetric)
    zoom_intensity: factor to multiply the focal length at max zoom
    """
    import copy
    traj = generate_path(viewpoint_cameras, n_frames=n_frames)
    # Get original focal length from the first camera
    cam0 = viewpoint_cameras[0]
    orig_fovx = cam0.FoVx
    orig_fovy = cam0.FoVy
    orig_focalx = cam0.image_width / (2 * np.tan(orig_fovx / 2))
    orig_focaly = cam0.image_height / (2 * np.tan(orig_fovy / 2))
    # Compute new focal for each frame
    for i, cam in enumerate(traj):
        cam = copy.deepcopy(cam)
        # Zoom in
        if zoom_start <= i < zoom_start + zoom_duration:
            t = (i - zoom_start) / max(zoom_duration - 1, 1)
            zoom_factor = 1 + t * (zoom_intensity - 1)
        # Zoom out
        elif zoom_start + zoom_duration <= i < zoom_start + 2 * zoom_duration:
            t = (i - (zoom_start + zoom_duration)) / max(zoom_duration - 1, 1)
            zoom_factor = zoom_intensity - t * (zoom_intensity - 1)
        else:
            zoom_factor = 1.0
        # Update focal length and FoV
        new_focalx = orig_focalx * zoom_factor
        new_focaly = orig_focaly * zoom_factor
        new_fovx = 2 * np.arctan(cam.image_width / (2 * new_focalx))
        new_fovy = 2 * np.arctan(cam.image_height / (2 * new_focaly))
        cam.FoVx = new_fovx
        cam.FoVy = new_fovy
        # Update projection matrix
        from utils.graphics_utils import getProjectionMatrix
        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=new_fovx, fovY=new_fovy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        traj[i] = cam
    return traj

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
    args = get_combined_args(parser)
    print("Creating video for " + args.model_path)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    triangles = TriangleModel(dataset.sh_degree)

    triangles.load_parameters(os.path.join(args.model_path, "point_cloud/iteration_30000"), segment=False)

    triangles.scaling = 4


    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=args.iteration,
                  shuffle=False)



    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    traj_dir = os.path.join(args.model_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    render_path = os.path.join(traj_dir, "renders")
    os.makedirs(render_path, exist_ok=True)
    
    n_frames = 240*5
    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)

    
    with torch.no_grad():
        for idx, view in enumerate(tqdm(cam_traj, desc="Rendering progress")):
            rendering = render(view, triangles, pipe, background)
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering["render"], os.path.join(traj_dir, "renders", '{0:05d}'.format(idx) + ".png"))

            """render_normal  = rendering['surf_normal']
            render_normal_np = render_normal.cpu().detach().numpy()
            global_min = render_normal_np.min()
            global_max = render_normal_np.max()
            render_normal_np = (render_normal_np - global_min) / (global_max - global_min)
            render_normal_np = np.transpose(render_normal_np, (1, 2, 0))  # HWC format
            render_normal_np = (render_normal_np * 255).astype(np.uint8)
            image_normal = Image.fromarray(render_normal_np)
            plt.imsave(os.path.join(traj_dir, "renders", '{0:05d}'.format(idx) + ".png"), image_normal)"""
            
    image_folder = os.path.join(traj_dir, "renders")
    output_video = args.save_as + '.mp4'

    # Get all image files sorted by name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Create video writer (FPS = 30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    # Write each image to the video
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()

    print(f'Video saved as {output_video}')

    # python create_video.py -m "/gpfs/home/acad/ulg-intelsig/jheld/triangle-splatting-mesh/trianglesplatting++/Treehill/best" -s /gpfs/scratch/acad/telim/datasets/MipNeRF360/treehill/ --eval -i images_4
    
    # python create_video.py -m "/gpfs/home/acad/ulg-intelsig/jheld/triangle-splatting-mesh/trianglesplatting++/Truck/anti aliasing 20k" -s /gpfs/scratch/acad/telim/datasets/tandt/truck/ --eval -i images_4