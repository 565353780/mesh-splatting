import sys
sys.path.append('../camera-control')

import os
import uuid
import torch
import lpips
import torch.nn.functional as F

from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter

from fused_ssim import fused_ssim

from triangle_renderer import render
from scene import Scene, TriangleModel
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor

from camera_control.Method.io import loadMeshFile


def training(
        dataset,
        opt,
        pipe,
        checkpoint,
        debug_from,
        scene_name,
        ):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Validate mesh file is provided
    if not dataset.mesh_file or dataset.mesh_file == "":
        raise ValueError("mesh_file is required! Please provide a valid mesh file path using --mesh_file argument.")

    # Load mesh from file
    print("Loading mesh from", dataset.mesh_file)
    mesh = loadMeshFile(dataset.mesh_file)

    if mesh is None:
        raise ValueError(f"Failed to load mesh from {dataset.mesh_file}")

    triangles = TriangleModel(dataset.sh_degree)
    triangles.load_mesh(mesh)

    # Validate mesh was loaded correctly
    if triangles.vertices.shape[0] == 0:
        raise ValueError("Mesh has no vertices!")
    if triangles._triangle_indices.shape[0] == 0:
        raise ValueError("Mesh has no faces!")

    print(f"Mesh loaded successfully:")
    print(f"  - Vertices: {triangles.vertices.shape[0]}")
    print(f"  - Faces: {triangles._triangle_indices.shape[0]}")
    print(f"  - SH degree: {triangles.active_sh_degree}")

    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)

    # Setup optimizer for vertex positions only (fixed topology mode)
    # In fix_topology mode, only vertex positions are optimized
    # RGB and opacity are fixed at 1.0 and not optimized
    print("Optimizing only vertex positions (fixed topology).")
    print("RGB and opacity are fixed at 1.0 and will not be optimized.")
    triangles.optimizer = torch.optim.Adam(
        [{'params': [triangles.vertices], 'lr': opt.lr_triangles_points_init, "name": "vertices"}],
        lr=0.0, eps=1e-15)

    # Disable gradients for other parameters to ensure only vertices are optimized
    # RGB features (features_dc and features_rest) are fixed
    triangles._features_dc.requires_grad_(False)
    triangles._features_rest.requires_grad_(False)
    # Opacity (vertex_weight) is fixed at 1.0 and not optimized
    triangles.vertex_weight.requires_grad_(False)

    triangles.triangle_scheduler_args = get_expon_lr_func(
        lr_init=opt.lr_triangles_points_init,
        lr_final=opt.lr_triangles_points_init/100,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Sigma schedule parameters
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)

    for iteration in range(first_iter, opt.iterations + 1):

        # Supersampling
        if iteration == opt.start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == opt.start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Sigma schedule
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, "normal_map", None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]
            seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]
            gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)

        # FINAL LOSS
        loss = loss_image

        # Vertex depth regularization
        Lvertex_depth_pure = 0.0
        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            depth_down = render_pkg["surf_depth"]
            vertex_depth_out = render_pkg["vertex_depth_out"]
            image_2D = render_pkg["image_2D"]
            vertex_rendered = render_pkg["vertex_rendered"]
            Lvertex_depth_pure = vertex_depth_loss_hr(
                vertex_depth_out,
                image_2D,
                vertex_rendered,
                depth_down,
                max_diff_threshold=opt.max_diff_threshold,
            )
            Lvertex_depth = lambda_vertex * Lvertex_depth_pure
            loss += Lvertex_depth
        else:
            Lvertex_depth = 0

        # Depth loss
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "invdepthmap", None) is not None:
            invDepth = 1.0 / (render_pkg["expected_depth"] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
        else:
            Ll1depth = 0

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        # Normal regularization (2DGS)
        Lnormal_pure = 0.0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            Lnormal_pure = normal_error.mean()
            Lnormal = lambda_normal * Lnormal_pure
            loss += Lnormal
        else:
            Lnormal = 0

        # supervised normal loss
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
            normal_loss_super = lambda_normals_super * (normal_error).mean()
            loss += normal_loss_super

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), scene, render, (pipe, background))

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)

    scene.save(iteration)
    print("Training is done")
    print(f"Final mesh statistics:")
    print(f"  - Vertices: {triangles.vertices.shape[0]}")
    print(f"  - Faces: {triangles._triangle_indices.shape[0]}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, scene_name, iteration, pixel_loss, loss, loss_fn, elapsed, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report samples of training set
    if iteration % 1000 == 0 or iteration == 1:
        torch.cuda.empty_cache()
        config = {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}

        if config['cameras'] and len(config['cameras']) > 0:
            pixel_loss_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            total_time = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                end_event.record()
                torch.cuda.synchronize()
                runtime = start_event.elapsed_time(end_event)
                total_time += runtime

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == 1:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                pixel_loss_test += loss_fn(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                lpips_test += lpips_fn(image, gt_image).mean().double()
            psnr_test /= len(config['cameras'])
            pixel_loss_test /= len(config['cameras'])       
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])  
            total_time /= len(config['cameras'])
            fps = 1000.0 / total_time
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} FPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test, fps))

            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--wandb_name', default="Test", type=str)
    parser.add_argument('--scene_name', default="Garden", type=str)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps,
             ops,
             pps,
             args.start_checkpoint,
             args.debug_from,
             args.scene_name,
             )
    
    # All done
    print("\nTraining complete.")
