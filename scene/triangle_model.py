#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import trimesh
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
import math
from simple_knn._C import distCUDA2
import math
import rdel



def random_rotation_matrices(num_matrices, device='cpu'):
    """
    Returns a tensor of shape (num_matrices, 3, 3) containing 
    random 3D rotation matrices.
    """
    axis = torch.randn(num_matrices, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    
    angles = 2.0 * math.pi * torch.rand(num_matrices, device=device)
    sin_t = torch.sin(angles)
    cos_t = torch.cos(angles)
    
    K = torch.zeros(num_matrices, 3, 3, device=device)
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1] = -uz
    K[:, 0, 2] =  uy
    K[:, 1, 0] =  uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] =  ux

    K2 = K.bmm(K)

    I = torch.eye(3, device=device).unsqueeze(0).expand(num_matrices, -1, -1)

    sin_term = sin_t.view(-1, 1, 1) * K
    cos_term = (1.0 - cos_t).view(-1, 1, 1) * K2

    return I + sin_term + cos_term


def fibonacci_directions(nb_points, device='cpu'):
    """
    Generate nb_points points on the unit sphere using a Fibonacci approach.
    Returns a tensor of shape (nb_points, 3).
    """
    directions = []
    for i in range(nb_points):
        z_coord = 1.0 - (2.0 * i / (nb_points - 1))
        z_coord = torch.tensor(z_coord, device=device)
        radius_xy = torch.sqrt(1.0 - z_coord * z_coord)
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i

        x_unit = radius_xy * torch.cos(torch.tensor(theta, device=device))
        y_unit = radius_xy * torch.sin(torch.tensor(theta, device=device))

        directions.append(torch.stack([x_unit, y_unit, z_coord]))
    return torch.stack(directions, dim=0)


def generate_triangles_in_chunks(x, y, z, radii, nb_points=3, chunk_size=2000):
    device = x.device

    num_centers = x.shape[0]

    base_dirs = fibonacci_directions(nb_points, device=device)
    out_points = torch.zeros(num_centers, nb_points, 3, device=device)

    for start_idx in range(0, num_centers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_centers)

        x_chunk = x[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        z_chunk = z[start_idx:end_idx]
        r_chunk = radii[start_idx:end_idx]

        chunk_size_actual = x_chunk.shape[0]

        R_chunk = random_rotation_matrices(chunk_size_actual, device=device)

        for i in range(nb_points):
            dir_i = base_dirs[i]

            dir_i_expanded = dir_i.view(1, 3, 1).expand(chunk_size_actual, -1, -1)

            rotated = R_chunk.bmm(dir_i_expanded)
            rotated = rotated.squeeze(-1)

            scaled = rotated * r_chunk.view(-1, 1)

            centers = torch.stack([x_chunk, y_chunk, z_chunk], dim=1)

            result_pts = centers + scaled

            out_points[start_idx:end_idx, i, :] = result_pts

    return out_points


class TriangleModel:

    def setup_functions(self):
        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int, use_sparse_adam : bool = False):

        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = 0
        self.pixel_count = 0
        self.importance_score = 0
        self.add_percentage = 1.0

        self.scaling = 1

        self.laplacian_update_freq = 50  # Update every 50 iterations
        self.iteration_count = 0

        self.use_sparse_adam = use_sparse_adam

        self.setup_functions()

    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size
        point_cloud_state_dict["pixel_count"] = self.pixel_count

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))


    def load_mesh(self, mesh: trimesh.Trimesh, device="cuda", active_sh_degree=3, assume_yup_to_zup=False, training_args=None):
        """
        Load vertices, faces, and SH features from a PLY file into the current object.
        Fields not derivable from the PLY, like vertex_weight, sigma, importance_score, are ignored.
        """
        SH_C0 = 0.28209479177387814

        def _to_float01(colors_np):
            if colors_np is None:
                return None
            if colors_np.ndim != 2 or colors_np.shape[1] < 3:
                return None
            rgb = colors_np[:, :3]
            if rgb.dtype == np.uint8:
                return rgb.astype(np.float32) / 255.0
            return rgb.astype(np.float32)

        verts_np = mesh.vertices.astype(np.float32).copy()
        if assume_yup_to_zup:
            # If your PLY is Y-up and you want Z-up internally, apply inverse of (x, y, z)->(x, z, -y)
            y = verts_np[:, 1].copy()
            z = verts_np[:, 2].copy()
            verts_np[:, 1] = -z
            verts_np[:, 2] = y

        faces_np = mesh.faces.astype(np.int32).copy() if mesh.faces is not None else np.empty((0, 3), np.int32)

        colors01 = None
        if getattr(mesh, "visual", None) is not None and hasattr(mesh.visual, "vertex_colors"):
            colors01 = _to_float01(mesh.visual.vertex_colors)

        V = int(verts_np.shape[0])
        verts = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
        faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)

        # features_dc: initialize to white (RGB=1.0) which maps to f_dc=(1.0-0.5)/SH_C0
        # Force RGB to 1.0 (white) regardless of mesh colors
        rgb_white = np.ones((V, 3), dtype=np.float32)  # RGB = [1.0, 1.0, 1.0]
        f_dc_rgb = ((rgb_white - 0.5) / SH_C0).clip(-4.0, 4.0).astype(np.float32)
        features_dc = torch.from_numpy(f_dc_rgb).to(device=device, dtype=torch.float32).unsqueeze(1).detach().clone().requires_grad_(True)

        # features_rest: zeros with shape [V, (deg+1)^2 - 1, 3]
        deg = int(active_sh_degree)
        num_coeff_total = (deg + 1) ** 2
        num_rest = max(0, num_coeff_total - 1)
        features_rest = torch.zeros((V, num_rest, 3), device=device, dtype=torch.float32, requires_grad=True)

        # Assign to object
        self.vertices = verts.requires_grad_(True)
        self._triangle_indices = faces
        self.active_sh_degree = deg
        self._features_dc = features_dc.requires_grad_(True)
        self._features_rest = features_rest.requires_grad_(True)

        # Initialize opacity to 1.0 (fully opaque)
        opacity_weight = 1.0
        self.opacity_floor = 0.9999
        vert_weight = inverse_sigmoid(opacity_weight * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))
        self._sigma = self.inverse_exponential_activation(0.0001)

        # Optional, quick report
        print(f"Loaded PLY: {mesh}")
        print(f"Vertices: {V}, Faces: {faces.shape[0]}, SH degree: {deg}, features_rest per color: {num_rest}")

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

        if training_args != None:
            self.optimizer = None
            self.triangle_scheduler_args = None
            param_groups = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self.vertices], 'lr': training_args.lr_triangles_points_init, "name": "vertices"},
                {'params': [self.vertex_weight], 'lr': training_args.weight_lr, "name": "vertex_weight"}
            ]
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

            self.triangle_scheduler_args = get_expon_lr_func(lr_init=training_args.lr_triangles_points_init,
                                                            lr_final=training_args.lr_triangles_points_init/100,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)



    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.75):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = state["active_sh_degree"]
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest      = state["features_rest"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone().requires_grad_(True)

        print("triangles: ", self._triangle_indices.shape)
        print("vertices: ", self.vertices.shape)

        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################

        self.opacity_floor = 0.999
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0016, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0016 / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': 0.0001, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property 
    def get_triangles_points(self): 
        return self._triangles

    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_vertices(self):
        return self.vertices

    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        # main features
        features_dc   = self._features_dc
        features_rest = self._features_rest
        feats_main = torch.cat((features_dc, features_rest), dim=1)  # [Vmain, F, 3]
        return feats_main
       
    @property
    def get_vertex_weight(self):
        main_w = self.opacity_activation(self.vertex_weight)
        return main_w

    def oneupSHdegree(self):
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, opacity : float, set_sigma : float):

        init_size = 2.23
        nb_points = 3  # 3 verts per triangle

        # --- Load PCD ---
        pcd_points = np.asarray(pcd.points)            # [N,3] (CPU, np)
        pcd_colors = np.asarray(pcd.colors)            # [N,3] (CPU, np)

        fused_point_cloud = torch.tensor(pcd_points, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_rgb   = torch.tensor(pcd_colors, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_sh    = RGB2SH(fused_color_rgb)                                       # [N,3]

        # SH features per *vertex* will be built after expansion to 3N
        # but we keep your original features layout
        base_feat_dim = (self.max_sh_degree + 1) ** 2

        # --- Scene size (same logic) ---
        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2]
        width  = x.max() - x.min()
        height = y.max() - y.min()
        depth  = z.max() - z.min()
        scene_size = torch.max(torch.stack([width, height, depth]))
        if scene_size.item() > 300:
            print("Scene is large, we increase the threshold")
            self.large = True

        # --- Per-point radii using your GPU NN distance (distCUDA2 returns squared NN dist) ---
        total_points = pcd_points  # naming for clarity
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()),
            1e-7
        )  # [N]
        radii = init_size * torch.sqrt(dist2).unsqueeze(1)  # [N,1]

        # --- Create 1 independent triangle per point (returns [N,3,3]) ---
        points_per_triangle = generate_triangles_in_chunks(x, y, z, radii, nb_points=nb_points)  # [N,3,3]

        # --- Flatten to vertex buffer and build triangle indices ---
        N = fused_point_cloud.shape[0]   # number of triangles
        _points = points_per_triangle.reshape(N * nb_points, 3).contiguous()  # [3N,3]
        faces = torch.arange(N * nb_points, device=_points.device, dtype=torch.int64).view(N, nb_points)  # [N,3]
        faces = faces.to(torch.int32)  # match your Delaunay path dtype

        # --- Per-vertex SH features (color from source point, repeated to its 3 verts) ---
        per_vertex_color_sh = fused_color_sh.repeat_interleave(nb_points, dim=0)  # [3N,3]
        features = torch.zeros((per_vertex_color_sh.shape[0], 3, base_feat_dim),
                            dtype=torch.float32, device="cuda")                 # [3N,3,F]
        features[:, :3, 0] = per_vertex_color_sh
        # features[:, 3:, 1:] stays zero (no higher SH bands initialized)

        # --- Parameters aligned with your Delaunay initializer ---
        self.vertices = nn.Parameter(_points.requires_grad_(True))                 # [3N,3]
        self._triangle_indices = faces                                            # [N,3] int32

        vert_weight = inverse_sigmoid(
            opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float32, device="cuda")
        )
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))        # [3N,1]

        # solid triangles
        self._sigma = self.inverse_exponential_activation(set_sigma)

        # SH feature tensors (transpose to [3N, 1, F] and [3N, (3-1)=2, F] like your code)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        # Per-triangle buffers (match Delaunay sizing by triangles count)
        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

      
  
    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):
      
        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    
    def update_learning_rate_delaunay(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = self.triangle_scheduler_args(iteration)
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles):
        # Create dictionary of new tensors to append
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        
        # Append new tensors to optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # Update model parameters
        self.vertices = optimizable_tensors["vertices"]
        self.vertex_weight = optimizable_tensors["vertex_weight"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        # Update triangle indices
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")



    def _update_params_fast(self, selected_indices, iteration):
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_ab, m_bc])
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0
        
        opacity_u = self.opacity_activation(self.vertex_weight[u])
        opacity_v = self.opacity_activation(self.vertex_weight[v])
        avg_opacity = (opacity_u + opacity_v) / 2.0
        avg_opacity = torch.clamp(avg_opacity, self.opacity_floor + self.eps, 1 - self.eps)
        new_vertex_weight = self.inverse_opacity_activation(avg_opacity)

        new_triangles = subdivided_triangles
        
        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # Prune optimizer state
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor


    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]
            if isinstance(self.pixel_count, torch.Tensor) and self.pixel_count.numel() > 0:
                self.pixel_count = self.pixel_count[valid_tris]
            

        # Prune vertex-related parameters using the initial mask
        self._prune_vertex_optimizer(vertex_mask)

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._prune_vertex_optimizer(mask_referenced)

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask]
        self._triangle_indices = self._triangle_indices.to(torch.int32)
        self.image_size = self.image_size[mask]
        self.importance_score = self.importance_score[mask]
        self.pixel_count = self.pixel_count[mask]
        

    def _sample_alives(self, probs, num, alive_indices=None):
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        return sampled_idxs        

    def add_new_gs(self, iteration, cap_max, splitt_large_triangles):

        current_num_points = self.vertices.shape[0]
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Find indexes based on proba
        triangle_transp = self.importance_score
        probs = triangle_transp.squeeze()

        areas = self.triangle_areas().squeeze()
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(self.image_size < self.size_probs_zero_image_space, torch.zeros_like(probs), probs) # dont splitt if smaller than 10

        rand_idx = self._sample_alives(probs=probs, num=num_gs)

        # Split the largest triangles
        split_large = splitt_large_triangles
        k = min(split_large, areas.numel())  
        _, top_idx = torch.topk(areas, k, largest=True, sorted=False)

        # 3) combine and deduplicate
        add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)

        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles) = self._update_params_fast(add_idx, iteration)

        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles)

        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        mask[add_idx] = False
        self.prune_triangles(mask)



    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True):
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            mask = self.vertices.shape[0]
            y = self.get_vertex_weight[:mask].detach()
            y = y.clamp(new_m + self.eps, 1.0 - self.eps)   # clamp to the *new* floor
        self.opacity_floor = new_m
        new_logits = self.inverse_opacity_activation(y)
        with torch.no_grad():
            self.vertex_weight.data.copy_(new_logits)


    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas


    
    def run_restricted_delaunay(self):

        print("Running restricted delaunay... for ", self.vertices.shape[0], " vertices.")

        self._triangle_indices = self._triangle_indices.detach().cpu().numpy()

        faces_ = rdel.run(
            self.vertices.detach().cpu().numpy(),
            self._triangle_indices,
            verbose=False,  # print timings and extra logs if True
            orient=False    # try to consistently orient face normals if True
        )

        self._triangle_indices = torch.as_tensor(np.asarray(faces_, dtype=np.int64), device='cuda').contiguous()
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        print("We have after re-delaunay ", self._triangle_indices.shape[0], " triangles.")

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
