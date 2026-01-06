/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

 #include "backward.h"
 #include "auxiliary.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;
 
 __device__ void circumcenter_backward(float3 A, float3 B, float3 C, float3 dU, float3* dA, float3* dB, float3* dC) {
    // Forward recomputation
    float3 AB = make_float3(B.x-A.x, B.y-A.y, B.z-A.z);
    float3 AC = make_float3(C.x-A.x, C.y-A.y, C.z-A.z);
    
    float3 N = make_float3(
        AB.y*AC.z - AB.z*AC.y,
        AB.z*AC.x - AB.x*AC.z,
        AB.x*AC.y - AB.y*AC.x
    );
    
    float AB2 = dot_float3(AB, AB);
    float AC2 = dot_float3(AC, AC);
    float denom = 2.0f * dot_float3(N, N);
    float inv_denom = 1.0f / denom;
    
    // Backward pass
    float3 term_sum = make_float3(
        (N.y*AB.z - N.z*AB.y)*AC2 + (AC.y*N.z - AC.z*N.y)*AB2,
        (N.z*AB.x - N.x*AB.z)*AC2 + (AC.z*N.x - AC.x*N.z)*AB2,
        (N.x*AB.y - N.y*AB.x)*AC2 + (AC.x*N.y - AC.y*N.x)*AB2
    );
    
    // Gradient through U = term_sum / denom
    float3 dterm_sum = scale_float3(dU, inv_denom);
    float ddenom = -dot_float3(term_sum, dU) / (denom*denom);
    
    // Intermediate gradients
    float dAB2 = 0.0f, dAC2 = 0.0f;
    float3 dN = make_float3(0,0,0);
    float3 dAB = make_float3(0,0,0);
    float3 dAC = make_float3(0,0,0);
    
    // Backprop through term_sum components
    for (int i = 0; i < 2; i++) {
        float3 cross_vec = (i == 0) ? 
            make_float3(N.y*AB.z - N.z*AB.y, N.z*AB.x - N.x*AB.z, N.x*AB.y - N.y*AB.x) :
            make_float3(AC.y*N.z - AC.z*N.y, AC.z*N.x - AC.x*N.z, AC.x*N.y - AC.y*N.x);
            
        float scalar = (i == 0) ? AC2 : AB2;
        float3 dcross_vec = scale_float3(dterm_sum, scalar);
        float dscalar = dot_float3(dterm_sum, cross_vec);
        
        if (i == 0) dAC2 += dscalar;
        else dAB2 += dscalar;
        
        // Cross product gradients
        if (i == 0) {
            dN = add_float3(dN, cross_float3(AB, dcross_vec));
            dAB = add_float3(dAB, cross_float3(dcross_vec, N));
        } else {
            dAC = add_float3(dAC, cross_float3(N, dcross_vec));
            dN = add_float3(dN, cross_float3(dcross_vec, AC));
        }
    }
    
    // Gradient through |N|^2
    dN = add_float3(dN, scale_float3(N, 2.0f * ddenom * 2.0f));
    
    // Gradient through N = cross(AB, AC)
    dAB = add_float3(dAB, cross_float3(AC, dN));
    dAC = add_float3(dAC, cross_float3(dN, AB));
    
    // Gradients through squared lengths
    dAB = add_float3(dAB, scale_float3(AB, 2.0f * dAB2));
    dAC = add_float3(dAC, scale_float3(AC, 2.0f * dAC2));
    
    // Final vertex gradients
    *dA = add_float3(scale_float3(dAB, -1.0f), scale_float3(dAC, -1.0f));
    *dB = dAB;
    *dC = dAC;
	*dA = add_float3(*dA, dU);
}
 
 // Backward pass for conversion of spherical harmonics to RGB for
 // each Triangle.
 __device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dvertices3D, glm::vec3* dL_dshs)
 {
	 // Compute intermediate values, as it is done during forward
	 glm::vec3 pos = means;
	 glm::vec3 dir_orig = pos - campos;
	 glm::vec3 dir = dir_orig / glm::length(dir_orig);
 
	 glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
 
	 // Use PyTorch rule for clamping: if clamping was applied,
	 // gradient becomes 0.
	 glm::vec3 dL_dRGB = dL_dcolor[idx];
	 //printf("Compute color for vertex %d with gradient: %.10f, %.10f, %.10f \n", idx, dL_dRGB.x, dL_dRGB.y, dL_dRGB.z);
	 dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	 dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	 dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

 
	 glm::vec3 dRGBdx(0, 0, 0);
	 glm::vec3 dRGBdy(0, 0, 0);
	 glm::vec3 dRGBdz(0, 0, 0);
	 float x = dir.x;
	 float y = dir.y;
	 float z = dir.z;
 
	 // Target location for this Triangle to write SH gradients to
	 glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;
 
	 // No tricks here, just high school-level calculus.
	 float dRGBdsh0 = SH_C0;
	 dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	 if (deg > 0)
	 {
		 float dRGBdsh1 = -SH_C1 * y;
		 float dRGBdsh2 = SH_C1 * z;
		 float dRGBdsh3 = -SH_C1 * x;
		 dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		 dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		 dL_dsh[3] = dRGBdsh3 * dL_dRGB;
 
		 dRGBdx = -SH_C1 * sh[3];
		 dRGBdy = -SH_C1 * sh[1];
		 dRGBdz = SH_C1 * sh[2];
 
		 if (deg > 1)
		 {
			 float xx = x * x, yy = y * y, zz = z * z;
			 float xy = x * y, yz = y * z, xz = x * z;
 
			 float dRGBdsh4 = SH_C2[0] * xy;
			 float dRGBdsh5 = SH_C2[1] * yz;
			 float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			 float dRGBdsh7 = SH_C2[3] * xz;
			 float dRGBdsh8 = SH_C2[4] * (xx - yy);
			 dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			 dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			 dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			 dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			 dL_dsh[8] = dRGBdsh8 * dL_dRGB;
 
			 dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			 dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			 dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];
 
			 if (deg > 2)
			 {
				 float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				 float dRGBdsh10 = SH_C3[1] * xy * z;
				 float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				 float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				 float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				 float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				 float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				 dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				 dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				 dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				 dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				 dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				 dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				 dL_dsh[15] = dRGBdsh15 * dL_dRGB;
 
				 dRGBdx += (
					 SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					 SH_C3[1] * sh[10] * yz +
					 SH_C3[2] * sh[11] * -2.f * xy +
					 SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					 SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					 SH_C3[5] * sh[14] * 2.f * xz +
					 SH_C3[6] * sh[15] * 3.f * (xx - yy));
 
				 dRGBdy += (
					 SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					 SH_C3[1] * sh[10] * xz +
					 SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					 SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					 SH_C3[4] * sh[13] * -2.f * xy +
					 SH_C3[5] * sh[14] * -2.f * yz +
					 SH_C3[6] * sh[15] * -3.f * 2.f * xy);
 
				 dRGBdz += (
					 SH_C3[1] * sh[10] * xy +
					 SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					 SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					 SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					 SH_C3[5] * sh[14] * (xx - yy));
			 }
		 }
	 }

 }
 

 __global__ void computeVertexColorsCUDA(
    int V, int D, int M,
	int W, int H,
	const float* viewmatrix,
	const float* proj,
	const float* vertices,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	const float* dL_dcolor,
	const float* dL_dpoints2D,
	glm::vec3* dL_dvertices3D,
	float* dL_dsh,
	const float* dL_dvertice_depth)
{	
	
    auto idx = cg::this_grid().thread_rank();
    if (idx >= V)
        return;


	glm::vec3 vertex {
		vertices[3 * idx + 0],
		vertices[3 * idx + 1],
		vertices[3 * idx + 2]
	};

	// back‑prop its own color→SH and color→position
	computeColorFromSH(idx, D, M, vertex, *campos,
		shs,
		clamped,
		(glm::vec3*)dL_dcolor,   
		(glm::vec3*)dL_dvertices3D, 
		(glm::vec3*)dL_dsh
	);

	float3 dL_ddepht = {0.0f, 0.0f, dL_dvertice_depth[idx]};
	float3 transposed_dL_ddepth = transformPoint4x3Transpose(dL_ddepht, viewmatrix);

	dL_dvertices3D[idx].x = transposed_dL_ddepth.x;
	dL_dvertices3D[idx].y = transposed_dL_ddepth.y;
	dL_dvertices3D[idx].z = transposed_dL_ddepth.z;

}

 
// Backward pass of the preprocessing steps, except
 // for the covariance computation and inversion
 // (those are handled by a previous kernel call)
 template<int C>
 __global__ void preprocessCUDA(
	 int P, int D, int M,
	 const float* vertices,
	 const int* triangles_indices,
	 const float* vertex_weights,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* proj,
	 const float* viewmatrix,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const glm::vec3* campos,
	 glm::vec3* dL_dvertices3D,
	 float* dL_dvertice_weights,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh)
 {
	 auto idx = cg::this_grid().thread_rank();

	 if (idx >= P || !(radii[idx] > 0))
		 return;


	 const int cumsum_for_triangle = 3 * idx;;
	 const int offset = 3 * cumsum_for_triangle;
	 float3 center_triangle = {0.0f, 0.0f, 0.0f};
	 float sum_x[MAX_NB_POINTS] = {0.0f};
	 float sum_y[MAX_NB_POINTS] = {0.0f};
	 float sum_z[MAX_NB_POINTS] = {0.0f};

	 float min_weight = INFINITY;
	 int id_lowest_weight = -1;
	 for (int i = 0; i < 3; i++) {
		int vertex_index = triangles_indices[cumsum_for_triangle + i];

		center_triangle.x += vertices[3 * vertex_index];
		center_triangle.y += vertices[3 * vertex_index + 1];
		center_triangle.z += vertices[3 * vertex_index + 2];

		float weight = vertex_weights[vertex_index];

		if (weight < min_weight) {
			id_lowest_weight = vertex_index;
			min_weight = weight;
		}
	 }
 
	 float3 total_sum = {center_triangle.x, center_triangle.y, center_triangle.z};
 
	 center_triangle.x /= 3;
	 center_triangle.y /= 3;
	 center_triangle.z /= 3;

	 float3 p_view_triangle;
	 if (!in_frustum_triangle(idx, center_triangle, viewmatrix, proj, false, p_view_triangle)){
		 return;
	 }

	 // Initialize loss accumulators for normals and offsets
	 float loss_points_x[MAX_NB_POINTS] = {0.0f};
	 float loss_points_y[MAX_NB_POINTS] = {0.0f};
 

	 for (int i = 0; i < 3; i++) {
		float dL_dnormal_x = dL_dnormals[cumsum_for_triangle + i].x;
		float dL_dnormal_y = dL_dnormals[cumsum_for_triangle + i].y;
		float dL_doffset = dL_doffsets[cumsum_for_triangle + i];

		float2 p1_conv = p_image[cumsum_for_triangle + i];
		float2 p2_conv = p_image[cumsum_for_triangle + (i + 1) % 3];

		// Calculate unnormalized normal components
		float nx = p2_conv.y - p1_conv.y;
		float ny = -(p2_conv.x - p1_conv.x);
		float norm = __fsqrt_rn(nx * nx + ny * ny);
		float inv_norm = 1.0f / norm;
	
		// Calculate normalized normal and offset
		float2 normal = {nx * inv_norm, ny * inv_norm};
		float offset = -(normal.x * p1_conv.x + normal.y * p1_conv.y);

		if (normal.x * points_xy_image[idx].x + normal.y * points_xy_image[idx].y + offset > 0) {
			dL_dnormal_x = -dL_dnormal_x;
			dL_dnormal_y = -dL_dnormal_y;
			dL_doffset = -dL_doffset;
		}

		// Add gradients from offset to normal gradients
		dL_dnormal_x += (-p1_conv.x) * dL_doffset;
		dL_dnormal_y += (-p1_conv.y) * dL_doffset;
	
		// Backprop through normalization
		float norm_sq = norm * norm;
		float inv_norm_cubed = 1.0f / (norm * norm_sq);
		float dL_dnx = (dL_dnormal_x * ny * ny - dL_dnormal_y * nx * ny) * inv_norm_cubed;
		float dL_dny = (-dL_dnormal_x * nx * ny + dL_dnormal_y * nx * nx) * inv_norm_cubed;
	
		// Compute gradients from unnormalized normal to points
		float2 dL_dp1_conv = {dL_dny, -dL_dnx}; // From ny: p1_conv.x, from nx: p1_conv.y
		float2 dL_dp2_conv = {-dL_dny, dL_dnx}; // From ny: p2_conv.x, from nx: p2_conv.y
	
		// Add gradients from offset to p1_conv
		dL_dp1_conv.x += (-normal.x) * dL_doffset;
		dL_dp1_conv.y += (-normal.y) * dL_doffset;

		loss_points_x[indices[cumsum_for_triangle + i]] += dL_dp1_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + i]] += dL_dp1_conv.y;

		loss_points_x[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.y;
	}
 
	 float3 dL_ddepht = {0.0f, 0.0f, dL_dmean2D[idx].x};
	 float3 transposed_dL_ddepth = transformPoint4x3Transpose(dL_ddepht, viewmatrix);

	 for (int i = 0; i < 3; i++) {
 
		int vertex_index = triangles_indices[cumsum_for_triangle + i];

		float mul1 = (proj[0] * vertices[3 * vertex_index] + proj[4] * vertices[3 * vertex_index + 1] + proj[8] * vertices[3 * vertex_index + 2] + proj[12]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
		float mul2 = (proj[1] * vertices[3 * vertex_index] + proj[5] * vertices[3 * vertex_index + 1] + proj[9] * vertices[3 * vertex_index + 2] + proj[13]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
				
		dL_dvertices3D[vertex_index].x += (proj[0] * p_w[cumsum_for_triangle + i] - proj[3] * mul1) * loss_points_x[i]  + (proj[1] * p_w[cumsum_for_triangle + i] - proj[3] * mul2) * loss_points_y[i] + transposed_dL_ddepth.x / 3;
		dL_dvertices3D[vertex_index].y += (proj[4] * p_w[cumsum_for_triangle + i] - proj[7] * mul1) * loss_points_x[i] + (proj[5] * p_w[cumsum_for_triangle + i] - proj[7] * mul2) * loss_points_y[i] + transposed_dL_ddepth.y / 3;
		dL_dvertices3D[vertex_index].z += (proj[8] * p_w[cumsum_for_triangle + i] - proj[11] * mul1) * loss_points_x[i] + (proj[9] * p_w[cumsum_for_triangle + i] - proj[11] * mul2) * loss_points_y[i] + transposed_dL_ddepth.z / 3;
		
	 }
 
	
	// Calculate the normal of the Triangle
	float3 normal_cvx = {0.0f, 0.0f, 0.0f};
	int vertex_index = triangles_indices[cumsum_for_triangle];
	float3 p0 = make_float3(
		vertices[3 * vertex_index + 0],
		vertices[3 * vertex_index + 1],
		vertices[3 * vertex_index + 2]
	);
	vertex_index = triangles_indices[cumsum_for_triangle + 1];
	float3 p1 = make_float3(
		vertices[3 * vertex_index + 0],
		vertices[3 * vertex_index + 1],
		vertices[3 * vertex_index + 2]
	);
	vertex_index = triangles_indices[cumsum_for_triangle + 2];
	float3 p2 = make_float3(
		vertices[3 * vertex_index + 0],
		vertices[3 * vertex_index + 1],
		vertices[3 * vertex_index + 2]
	);
	float3 v1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
	float3 v2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
	float3 v3 = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
 
	float3 unnorm_ross_prod = make_float3(
		 v1.y * v2.z - v1.z * v2.y,
		 v1.z * v2.x - v1.x * v2.z,
		 v1.x * v2.y - v1.y * v2.x
	);
	float3 cross_prod = transformVec4x3(unnorm_ross_prod, viewmatrix);
 
	float length = sqrtf(cross_prod.x*cross_prod.x + cross_prod.y*cross_prod.y + cross_prod.z*cross_prod.z);
	if (length > 1e-8f) {
		 cross_prod.x /= length;
		 cross_prod.y /= length;
		 cross_prod.z /= length;
	}
	normal_cvx = cross_prod;

	float3 p_view_triangle_ = transformPoint4x3(center_triangle, viewmatrix);
	// we normalize such that we have a unit vector and cos is between -1 and 1
	float length_viewpoint = sqrtf(p_view_triangle_.x*p_view_triangle_.x + p_view_triangle_.y*p_view_triangle_.y + p_view_triangle_.z*p_view_triangle_.z);
	length_viewpoint = max(length_viewpoint, 1e-4f);
	float3 normalized_camera_center;
	normalized_camera_center.x = p_view_triangle_.x / length_viewpoint;
	normalized_camera_center.y = p_view_triangle_.y / length_viewpoint;
	normalized_camera_center.z = p_view_triangle_.z / length_viewpoint;
 
	float3 dir = make_float3(
		 normalized_camera_center.x * normal_cvx.x,
		 normalized_camera_center.y * normal_cvx.y,
		 normalized_camera_center.z * normal_cvx.z
	);
	 
	float cos = -sumf3(dir);
	 
	const float threshold = 0.001f;
	if (fabsf(cos) < threshold) {
		return;
	}
		
	float multiplier = cos > 0 ? 1 : -1;
	normal_cvx = {cross_prod.x * multiplier, cross_prod.y * multiplier, cross_prod.z * multiplier};
 
	// ## BACKWARD
	float3 dL_dtn = {dL_dnormal3D[idx * 3 + 0]*multiplier, dL_dnormal3D[idx * 3 + 1]*multiplier, dL_dnormal3D[idx * 3 + 2]*multiplier};
 
	float matrix_w0[9], matrix_w1[9], matrix_w2[9];
 
	matrix_w0[0] = 0.0f;   matrix_w0[1] = -v3.z; matrix_w0[2] = v3.y;
	matrix_w0[3] = v3.z;   matrix_w0[4] = 0.0f;  matrix_w0[5] = -v3.x;
	matrix_w0[6] = -v3.y;  matrix_w0[7] = v3.x;  matrix_w0[8] = 0.0f;
 
	matrix_w1[0] = 0.0f;   matrix_w1[1] = v2.z;  matrix_w1[2] = -v2.y;
	matrix_w1[3] = -v2.z;  matrix_w1[4] = 0.0f;  matrix_w1[5] = v2.x;
	matrix_w1[6] = v2.y;   matrix_w1[7] = -v2.x; matrix_w1[8] = 0.0f;
 
	matrix_w2[0] = 0.0f;   matrix_w2[1] = -v1.z; matrix_w2[2] = v1.y;
	matrix_w2[3] = v1.z;   matrix_w2[4] = 0.0f;  matrix_w2[5] = -v1.x;
	matrix_w2[6] = -v1.y;  matrix_w2[7] = v1.x;  matrix_w2[8] = 0.0f;
 
	float normal_transpose[9];
	normal_transpose[0] = normal_cvx.x*normal_cvx.x; 
	normal_transpose[1] = normal_transpose[3] = normal_cvx.x*normal_cvx.y;
	normal_transpose[2] = normal_transpose[6] = normal_cvx.x*normal_cvx.z;
	normal_transpose[4] = normal_cvx.y*normal_cvx.y;
	normal_transpose[5] = normal_transpose[7] = normal_cvx.y*normal_cvx.z;
	normal_transpose[8] = normal_cvx.z*normal_cvx.z;
 
	float length_inv = 1 / length;
 
	float projection_matrix[9];
	projection_matrix[0] = viewmatrix[0];
	projection_matrix[1] = viewmatrix[4];
	projection_matrix[2] = viewmatrix[8];
	projection_matrix[3] = viewmatrix[1];
	projection_matrix[4] = viewmatrix[5];
	projection_matrix[5] = viewmatrix[9];
	projection_matrix[6] = viewmatrix[2];
	projection_matrix[7] = viewmatrix[6];
	projection_matrix[8] = viewmatrix[10];
 
	float matrix_w0_transformed[9], matrix_w1_transformed[9], matrix_w2_transformed[9];
	transformMat3x3(projection_matrix, matrix_w0, matrix_w0_transformed);
	transformMat3x3(projection_matrix, matrix_w1, matrix_w1_transformed);
	transformMat3x3(projection_matrix, matrix_w2, matrix_w2_transformed);
 
	float norm_times_matrix0[9], norm_times_matrix1[9], norm_times_matrix2[9];
	transformMat3x3(normal_transpose, matrix_w0_transformed, norm_times_matrix0);
	transformMat3x3(normal_transpose, matrix_w1_transformed, norm_times_matrix1);
	transformMat3x3(normal_transpose, matrix_w2_transformed, norm_times_matrix2);
 
	float matrix_substraction0[9], matrix_substraction1[9], matrix_substraction2[9];
	substractionMat3x3(matrix_w0_transformed, norm_times_matrix0, matrix_substraction0);
	substractionMat3x3(matrix_w1_transformed, norm_times_matrix1, matrix_substraction1);
	substractionMat3x3(matrix_w2_transformed, norm_times_matrix2, matrix_substraction2);
 
	float dL_dp0x = length_inv * matrix_substraction0[0] * dL_dtn.x + length_inv * matrix_substraction0[3] * dL_dtn.y + length_inv * matrix_substraction0[6] * dL_dtn.z;
	float dL_dp0y = length_inv * matrix_substraction0[1] * dL_dtn.x + length_inv * matrix_substraction0[4] * dL_dtn.y + length_inv * matrix_substraction0[7] * dL_dtn.z;
	float dL_dp0z = length_inv * matrix_substraction0[2] * dL_dtn.x + length_inv * matrix_substraction0[5] * dL_dtn.y + length_inv * matrix_substraction0[8] * dL_dtn.z;
 
	float dL_dp1x = length_inv * matrix_substraction1[0] * dL_dtn.x + length_inv * matrix_substraction1[3] * dL_dtn.y + length_inv * matrix_substraction1[6] * dL_dtn.z;
	float dL_dp1y = length_inv * matrix_substraction1[1] * dL_dtn.x + length_inv * matrix_substraction1[4] * dL_dtn.y + length_inv * matrix_substraction1[7] * dL_dtn.z;
	float dL_dp1z = length_inv * matrix_substraction1[2] * dL_dtn.x + length_inv * matrix_substraction1[5] * dL_dtn.y + length_inv * matrix_substraction1[8] * dL_dtn.z;
 
	float dL_dp2x = length_inv * matrix_substraction2[0] * dL_dtn.x + length_inv * matrix_substraction2[3] * dL_dtn.y + length_inv * matrix_substraction2[6] * dL_dtn.z;
	float dL_dp2y = length_inv * matrix_substraction2[1] * dL_dtn.x + length_inv * matrix_substraction2[4] * dL_dtn.y + length_inv * matrix_substraction2[7] * dL_dtn.z;
	float dL_dp2z = length_inv * matrix_substraction2[2] * dL_dtn.x + length_inv * matrix_substraction2[5] * dL_dtn.y + length_inv * matrix_substraction2[8] * dL_dtn.z;

	
	// Backpropagate gradients to vertices (Normal&Depth)
	vertex_index = triangles_indices[cumsum_for_triangle];
	vertex_index = triangles_indices[cumsum_for_triangle + 1];
	vertex_index = triangles_indices[cumsum_for_triangle + 2];

	atomicAdd(&dL_dvertice_weights[id_lowest_weight], dL_dopacity[idx]);


 }
 
 // Backward version of the rendering procedure.
 template <uint32_t C>
 __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
 renderCUDA(
	 const uint2* __restrict__ ranges,
	 const uint32_t* __restrict__ point_list,
	 int W, int H,
	 const float* __restrict__ bg_color,
	 const float sigma,
	 const int* __restrict__ triangles_indices,
	 const float2* __restrict__ normals,
	 const float* __restrict__ offsets,
	 const float4* __restrict__ conic_opacity,
	 const float* __restrict__ depths,
	 const float2* __restrict__ means2D,
	 const float2* __restrict__ phi_center,
	 const float2* __restrict__ p_image,
	 const float* __restrict__ vertex_depth,
	 const float* __restrict__ colors,
	 const float* __restrict__ final_Ts,
	 const uint32_t* __restrict__ n_contrib,
	 const float* __restrict__ dL_dpixels,
	 const float* __restrict__ dL_depths,
	 float2* __restrict__ dL_dnormals,
	 float* __restrict__ dL_doffsets,
	 float3* __restrict__ dL_dmean2D,
	 float* __restrict__ dL_dopacity,
	 float* __restrict__ dL_dnormal3D,
	 float* __restrict__ dL_dcolors,
	 float* __restrict__ dL_dpoints2D,
	 float* __restrict__ dL_dvertice_depth)
 {
	 // We rasterize again. Compute necessary block info.
	 auto block = cg::this_thread_block();
	 const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	 const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	 const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	 const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	 const uint32_t pix_id = W * pix.y + pix.x;
	 const float2 pixf = { (float)pix.x, (float)pix.y };
 
	 const bool inside = pix.x < W&& pix.y < H;
	 const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
 
	 const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
	 bool done = !inside;
	 int toDo = range.y - range.x;
 
	 __shared__ int collected_id[BLOCK_SIZE];
	 __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	 __shared__ float collected_colors[C * BLOCK_SIZE];
 
	 /*
	 ADDED FOR Triangle PURPOSES ==========================================================================
	 */
	 __shared__ float2 collected_p_images[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_depths[BLOCK_SIZE];
	 __shared__ float2 collected_xy[BLOCK_SIZE];
	 __shared__ float2 collected_phi_center[BLOCK_SIZE];
	 /*
	 ===================================================================================================
	 */
 
	 // In the forward, we stored the final value for T, the
	 // product of all (1 - alpha) factors. 
	 const float T_final = inside ? final_Ts[pix_id] : 0;
	 float T = T_final;
 
	 // We start from the back. The ID of the last contributing
	 // Triangle is known from each pixel from the forward.
	 uint32_t contributor = toDo;
	 const int last_contributor = inside ? n_contrib[pix_id] : 0;
 
	 float accum_rec[C] = { 0 };
	 float dL_dpixel[C];
	 if (inside)
		 for (int i = 0; i < C; i++)
			 dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];


	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
 
	 float last_alpha = 0;
	 float last_color[C] = { 0 };

 
	 // Traverse all triangles
	 for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	 {
		 // Load auxiliary data into shared memory, start in the BACK
		 // and load them in revers order.
		 block.sync();
		 const int progress = i * BLOCK_SIZE + block.thread_rank();
		 if (range.x + progress < range.y)
		 {
			 const int coll_id = point_list[range.y - progress - 1];
			 collected_id[block.thread_rank()] = coll_id;
			 collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			 for (int i = 0; i < C; i++)
				 collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			 collected_depths[block.thread_rank()] = depths[coll_id];
			 collected_xy[block.thread_rank()] = means2D[coll_id];
			 for (int k = 0; k < 3; k++) {
				collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[3 * coll_id + k];
				collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[3 * coll_id + k];;
				collected_p_images[MAX_NB_POINTS * block.thread_rank() + k] = p_image[3 * coll_id + k];
			}
			collected_phi_center[block.thread_rank()] = phi_center[coll_id];
		 }
		 block.sync();
 
		 // Iterate over triangles
		 for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		 {
 
			 contributor--;
			 if (contributor >= last_contributor)
				 continue;

			 int j_id = collected_id[j];
			 float4 con_o = collected_conic_opacity[j];
			 float normal[3] = {con_o.x, con_o.y, con_o.z};
			 float2 phi_center_min = collected_phi_center[j];
			 float distances[MAX_NB_POINTS];
			 float max_val = -INFINITY;
			 int base = j * MAX_NB_POINTS;
			 bool outside = false;
 
			 for (int k = 0; k < 3; k++) {
				 // Compute the current distance
				 float dist = (collected_normals[base + k].x * pixf.x
						  + collected_normals[base + k].y * pixf.y
						  + collected_offsets[base + k]);
				
				 if (dist > 0) {
					outside = true;
					break;
				 }
 
				 distances[k] = dist;
				 max_val = fmaxf(max_val, dist);
			 }

			 if (outside)
				 continue;
 
			 float phi_x = max_val;
			 float phi_final = phi_x * phi_center_min.x;
			 float Cx = fmaxf(0.0f,  __powf(phi_final, sigma));
 
			 const float alpha = min(0.99f, con_o.w * Cx);
 
			 if (alpha < 1.0f / 255.0f)
				 continue;
 
			 T = T / (1.f - alpha);
			 const float dchannel_dcolor = alpha * T;

			 float2 uv0 = collected_p_images[j * 3 + 0];
			 float2 uv1 = collected_p_images[j * 3 + 1];
			 float2 uv2 = collected_p_images[j * 3 + 2];

			 // vectors along the edges from uv0
			 float2 v0 = { uv1.x - uv0.x, uv1.y - uv0.y };
			 float2 v1 = { uv2.x - uv0.x, uv2.y - uv0.y };
			 // vector from uv0 to pixel
			 float2 v2 = { pixf.x  - uv0.x, pixf.y  - uv0.y };

			 // invert the 2×2 [v0 v1] matrix
			 float denom  = v0.x * v1.y - v1.x * v0.y;
			 float invDen = 1.0f / denom;    // assume non-degenerate

			 // barycentrics relative to uv0,uv1,uv2
			 float b0 = ( v2.x * v1.y - v1.x * v2.y) * invDen;
			 float b1 = (-v2.x * v0.y + v0.x * v2.y) * invDen;
			 float b2 = 1.0f - b0 - b1;

			 int aux = 3 * j_id;
			 int vertex_idx0 = triangles_indices[aux];
			 int vertex_idx1 = triangles_indices[aux + 1];
			 int vertex_idx2 = triangles_indices[aux + 2];

			 float depth_vertex_0 = vertex_depth[vertex_idx0];
			 float depth_vertex_1 = vertex_depth[vertex_idx1];
			 float depth_vertex_2 = vertex_depth[vertex_idx2];

			 float wA = b2;    // vertex0
			 float wB = b0;    // vertex1
			 float wC = b1;    // vertex2

			 // now blend them
			 float interp_color[C];
			 float sum0 = 0, sum1 = 0, sum2 = 0;
			 for (int ch = 0; ch < C; ++ch) {
				float dL_dcolor_ch = dchannel_dcolor * dL_dpixel[ch];
				float c0 = colors[vertex_idx0 * C + ch];
				float c1 = colors[vertex_idx1 * C + ch];
				float c2 = colors[vertex_idx2 * C + ch];

				interp_color[ch] = wA * c0 + wB * c1 + wC * c2;

				sum0 += dL_dcolor_ch * c0; // for db2
				sum1 += dL_dcolor_ch * c1; // for db0
				sum2 += dL_dcolor_ch * c2; // for db1
			 }

			 float dL_dalpha = 0.0f;
			 const int global_id = collected_id[j];
			 
			 for (int ch = 0; ch < C; ++ch) {
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = interp_color[ch];

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (interp_color[ch] - accum_rec[ch]) * dL_dchannel;

				int v0 = vertex_idx0;
				int v1 = vertex_idx1;
				int v2 = vertex_idx2;

				// backward (global‐vertex)
				float grad0 = dchannel_dcolor * dL_dchannel * b2;  // matches forward's c0 * b2
				float grad1 = dchannel_dcolor * dL_dchannel * b0;  // matches forward's c1 * b0
				float grad2 = dchannel_dcolor * dL_dchannel * b1;  // matches forward's c2 * b1

				atomicAdd(&dL_dcolors[v0*C + ch], grad0);
				atomicAdd(&dL_dcolors[v1*C + ch], grad1);
				atomicAdd(&dL_dcolors[v2*C + ch], grad2);

			 } 

			 float depth_interp = wA * depth_vertex_0 + wB * depth_vertex_1 + wC * depth_vertex_2;
			 float weight_here = alpha * T;
			 atomicAdd(&dL_dvertice_depth[vertex_idx0], dL_ddepth * weight_here * wA);
			 atomicAdd(&dL_dvertice_depth[vertex_idx1], dL_ddepth * weight_here * wB);
			 atomicAdd(&dL_dvertice_depth[vertex_idx2], dL_ddepth * weight_here * wC);

			 
			 // Backpropagation to the vertices
			 float dL_db0 = sum1;
		 	 float dL_db1 = sum2;
			 float dL_db2 = sum0;

			 if (contributor == median_contributor-1) {
				  //dL_dz += dL_dmedian_depth;
				  dL_db0 += dL_dmedian_depth * depth_vertex_1;
				  dL_db1 += dL_dmedian_depth * depth_vertex_2;
				  dL_db2 += dL_dmedian_depth * depth_vertex_0;

				  // Accumulate gradients to vertex depths
				  atomicAdd(&dL_dvertice_depth[vertex_idx0], dL_dmedian_depth * wA);
				  atomicAdd(&dL_dvertice_depth[vertex_idx1], dL_dmedian_depth * wB);
				  atomicAdd(&dL_dvertice_depth[vertex_idx2], dL_dmedian_depth * wC);
			  }

			 // Recompute necessary terms for derivatives
			 float denom_val = v0.x * v1.y - v1.x * v0.y;
			 float N0 = v2.x * v1.y - v1.x * v2.y;
			 float N1 = v0.x * v2.y - v2.x * v0.y;
			 float factor = invDen * invDen;

			 // Derivatives for vertex0 (uv0)
			 float dN0_dx0 = -v1.y + v2.y;
			 float dN0_dy0 = -v2.x + v1.x;
			 float dN1_dx0 = -v2.y + v0.y;
			 float dN1_dy0 = v2.x - v0.x;
			 float dD_dx0 = -v1.y + v0.y;
			 float dD_dy0 = -v0.x + v1.x;

			 float db0_dx0 = (dN0_dx0 * denom_val - N0 * dD_dx0) * factor;
			 float db0_dy0 = (dN0_dy0 * denom_val - N0 * dD_dy0) * factor;
			 float db1_dx0 = (dN1_dx0 * denom_val - N1 * dD_dx0) * factor;
			 float db1_dy0 = (dN1_dy0 * denom_val - N1 * dD_dy0) * factor;
			 float db2_dx0 = -db0_dx0 - db1_dx0;
			 float db2_dy0 = -db0_dy0 - db1_dy0;

			 float dL_dx0 = dL_db0 * db0_dx0 + dL_db1 * db1_dx0 + dL_db2 * db2_dx0;
			 float dL_dy0 = dL_db0 * db0_dy0 + dL_db1 * db1_dy0 + dL_db2 * db2_dy0;

			 // Derivatives for vertex1 (uv1)
			 float dN0_dx1 = 0;
			 float dN0_dy1 = 0;
			 float dN1_dx1 = v2.y;
			 float dN1_dy1 = -v2.x;
			 float dD_dx1 = v1.y;
			 float dD_dy1 = -v1.x;

			 float db0_dx1 = (dN0_dx1 * denom_val - N0 * dD_dx1) * factor;
			 float db0_dy1 = (dN0_dy1 * denom_val - N0 * dD_dy1) * factor;
			 float db1_dx1 = (dN1_dx1 * denom_val - N1 * dD_dx1) * factor;
			 float db1_dy1 = (dN1_dy1 * denom_val - N1 * dD_dy1) * factor;
			 float db2_dx1 = -db0_dx1 - db1_dx1;
			 float db2_dy1 = -db0_dy1 - db1_dy1;

			 float dL_dx1 = dL_db0 * db0_dx1 + dL_db1 * db1_dx1 + dL_db2 * db2_dx1;
			 float dL_dy1 = dL_db0 * db0_dy1 + dL_db1 * db1_dy1 + dL_db2 * db2_dy1;

			 // Derivatives for vertex2 (uv2)
			 float dN0_dx2 = -v2.y;
			 float dN0_dy2 = v2.x;
			 float dN1_dx2 = 0;
			 float dN1_dy2 = 0;
			 float dD_dx2 = -v0.y;
			 float dD_dy2 = v0.x;

			 float db0_dx2 = (dN0_dx2 * denom_val - N0 * dD_dx2) * factor;
			 float db0_dy2 = (dN0_dy2 * denom_val - N0 * dD_dy2) * factor;
			 float db1_dx2 = (dN1_dx2 * denom_val - N1 * dD_dx2) * factor;
			 float db1_dy2 = (dN1_dy2 * denom_val - N1 * dD_dy2) * factor;
			 float db2_dx2 = -db0_dx2 - db1_dx2;
			 float db2_dy2 = -db0_dy2 - db1_dy2;

			 float dL_dx2 = dL_db0 * db0_dx2 + dL_db1 * db1_dx2 + dL_db2 * db2_dx2;
			 float dL_dy2 = dL_db0 * db0_dy2 + dL_db1 * db1_dy2 + dL_db2 * db2_dy2;

			 // Update gradients for vertex positions
			 atomicAdd(&dL_dpoints2D[vertex_idx0 * 2], dL_dx0);
			 atomicAdd(&dL_dpoints2D[vertex_idx0 * 2 + 1], dL_dy0);
			 atomicAdd(&dL_dpoints2D[vertex_idx1 * 2], dL_dx1);
			 atomicAdd(&dL_dpoints2D[vertex_idx1 * 2 + 1], dL_dy1);
			 atomicAdd(&dL_dpoints2D[vertex_idx2 * 2], dL_dx2);
			 atomicAdd(&dL_dpoints2D[vertex_idx2 * 2 + 1], dL_dy2); 
 
			 // Propagate gradients w.r.t ray-splat depths
			 accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			 last_depth = depth_interp; // not collected_depths[j]
			 dL_dalpha += (depth_interp - accum_depth_rec) * dL_ddepth;
			 // Propagate gradients w.r.t. color ray-splat alphas
			 accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			 dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;
  
			 for (int ch = 0; ch < 3; ch++) {
				 accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				 last_normal[ch] = normal[ch];
				 dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				 atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			 }
 
			 dL_dalpha *= T;
			 // Update last alpha (to be used in the next iteration)
			 last_alpha = alpha;
 
			 // Account for fact that alpha also influences how much of
			 // the background color is added if nothing left to blend
			 float bg_dot_dpixel = 0;
			 for (int i = 0; i < C; i++)
				 bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			 dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
 
			 // Helpful reusable temporary variables
			 const float dL_dC = con_o.w * dL_dalpha;
			
			// Calculate gradient w.r.t phi_x 
			float dL_dphi_x = dL_dC * (sigma / phi_x) * Cx;
 
			 #pragma unroll
			 for (int k = 0; k < 3; k++) {
				if (fabsf(distances[k] - max_val) < 1e-6f) {
					float dL_dnormal_x = dL_dphi_x * pixf.x;
					float dL_dnormal_y = dL_dphi_x * pixf.y;
					atomicAdd(&(dL_dnormals[aux + k].x), dL_dnormal_x);
					atomicAdd(&(dL_dnormals[aux + k].y), dL_dnormal_y);
					atomicAdd(&(dL_doffsets[aux + k]), dL_dphi_x);
				}
			 }
 
			 // Update gradients w.r.t. opacity of the Triangle
			 atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * Cx);
 
		 }
	 }
 }
 
 void BACKWARD::preprocess(
	 int P, int D, int M,
	 const float* vertices,
	 const int* triangles_indices,
	 const float* vertex_weights,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* viewmatrix,
	 const float* projmatrix,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const float focal_x, float focal_y,
	 const float tan_fovx, float tan_fovy,
	 const glm::vec3* campos,
	 glm::vec3* dL_dvertices3D, 
	 float* dL_dvertice_weights,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh
	 )
 {
	 
	 // Propagate gradients for remaining steps: finish 3D mean gradients,
	 // propagate color gradients to SH (if desireD), propagate 3D covariance
	 // matrix gradients to scale and rotation.
	 preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		 P, D, M,
		 vertices,
		 triangles_indices,
		 vertex_weights,
		 W, H,
		 radii,
		 shs,
		 clamped,
		 projmatrix,
		 viewmatrix,
		 points_xy_image,
		 p_w,
		 p_image,
		 indices,
		 campos,
		 (glm::vec3*)dL_dvertices3D,
		 (float*) dL_dvertice_weights,
		 (float2*) dL_dnormals,
		 dL_doffsets,
		 (float3*)dL_dmean2D,
		 dL_dopacity,
		 dL_dnormal3D,
		 dL_dcolor,
		 dL_dsh
		 );
 }

 // Add this to the FORWARD namespace implementation
 void BACKWARD::computeVertexColorGradients(
    int V, int D, int M,
	int W, int H,
	const float* viewmatrix,
	const float* projmatrix,
	float* p_w,
	const float* vertices,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	const float* dL_dcolor,
	const float* dL_dpoints2D,
	glm::vec3* dL_dvertices3D,
	float* dL_dsh,
	const float* dL_dvertice_depth)
 {
    computeVertexColorsCUDA<<<(V + 255) / 256, 256>>>(
        V, D, M, W, H, viewmatrix, projmatrix, vertices, shs, clamped, campos, dL_dcolor, dL_dpoints2D, (glm::vec3*)dL_dvertices3D, dL_dsh, dL_dvertice_depth
    );
 }
 
 void BACKWARD::render(
	 const dim3 grid, const dim3 block,
	 const uint2* ranges,
	 const uint32_t* point_list,
	 int W, int H,
	 const float* bg_color,
	 const float sigma,
	 const int* triangles_indices,
	 const float2* normals,
	 const float* offsets,
	 const float4* conic_opacity,
	 const float* depths,
	 const float2* means2D,
	 const float2* phi_center,
	 const float2* p_image,
	 const float* vertex_depth,
	 const float* colors,
	 const float* final_Ts,
	 const uint32_t* n_contrib,
	 const float* dL_dpixels,
	 const float* dL_depths,
	 float2* dL_dnormals,
	 float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolors,
	 float* dL_dpoints2D,
	 float* dL_dvertice_depth
	)
 {
	 renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		 ranges,
		 point_list,
		 W, H,
		 bg_color,
		 sigma,
		 triangles_indices,
		 normals,
		 offsets,
		 conic_opacity,
		 depths,
		 means2D,
		 phi_center,
		 p_image,
		 vertex_depth,
		 colors,
		 final_Ts,
		 n_contrib,
		 dL_dpixels,
		 dL_depths,
		 dL_dnormals,
		 dL_doffsets,
		 dL_dmean2D,
		 dL_dopacity,
		 dL_dnormal3D,
		 dL_dcolors,
		 dL_dpoints2D,
		 dL_dvertice_depth
		 );
 }