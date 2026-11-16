
import torch    
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from estimotion_2part.models.loss import (
    ConsistencyLoss,
    chamfer_loss,
    entropy_loss,
    projection_matching_loss,
)


def _audit_trace_enabled():
    """Audit tracing switch (samuele/debug_audit/trace.py); False unless FREEARTGS_TRACE_ROOT is set."""
    import os
    return bool(os.environ.get("FREEARTGS_TRACE_ROOT"))

class TrajectoryDataset(Dataset):
    def __init__(self, source_points, target_points, w_ids, T_ids, knn_idx=None):
        self.source_points = source_points
        self.target_points = target_points
        self.w_ids = w_ids
        self.T_ids = T_ids
        self.knn_idx = knn_idx
        
    def __len__(self):
        return len(self.source_points)
    
    def __getitem__(self, idx):
        data = {
            'source_point': self.source_points[idx],
            'target_point': self.target_points[idx],
            'w_id': self.w_ids[idx],
            'T_id': self.T_ids[idx],
            'point_idx': idx
        }
        if self.knn_idx is not None:
            data['knn_indices'] = self.knn_idx[idx]
        return data

class Trans(nn.Module):
    def __init__(self, num_points, num_T, init_weights=None, init_T1=None, init_T2=None, cam_K=None, device="cuda"):
        super(Trans, self).__init__()
        # Initialize using member functions
        self.weight = self._initialize_weights(num_points, init_weights, device)
        self.T1, self.T2 = self._initialize_transform_matrices(num_T, init_T1, init_T2, device)
        self.cam_K = torch.tensor(cam_K, dtype=torch.float32, device=device)

    def _initialize_weights(self, num_points, init_weights=None, device="cuda"):
        """
        Initialize weight parameters
        Args:
            num_points: Number of points
            init_weights: Initial weight values, can be numpy array or None
            device: Device
        Returns:
            Weight parameter (nn.Parameter)
        """
        if init_weights is not None:
            init_weights = torch.tensor(init_weights, dtype=torch.float32, device=device)
            init_weights = torch.clamp(init_weights, 0.01, 0.99)
            return nn.Parameter(torch.logit(init_weights).unsqueeze(1))
        else:
            # Default initialization to 0.0 (logit space, corresponds to sigmoid(0)=0.5)
            return nn.Parameter(torch.full((num_points, 1), 0.0, device=device))

    def _initialize_transform_matrices(self, num_T, init_T1=None, init_T2=None, device="cuda"):
        """
        Initialize transformation matrix parameters
        Args:
            num_T: Number of transformation matrices
            init_T1: Initial T1 matrix, can be single 4x4 matrix, (num_T, 4, 4) array or None
            init_T2: Initial T2 matrix, can be single 4x4 matrix, (num_T, 4, 4) array or None
            device: Device
        Returns:
            T1 parameter, T2 parameter (nn.Parameter, nn.Parameter)
        """
        # Initialize T1
        if init_T1 is not None:
            if len(init_T1.shape) == 2:  # Single 4x4 matrix
                T1 = nn.Parameter(init_T1.unsqueeze(0).expand(num_T, -1, -1).clone())
            else:  # Already in (num_T, 4, 4) format
                T1 = nn.Parameter(torch.tensor(init_T1, dtype=torch.float32, device=device))
        else:
            # Default initialization to num_T identity matrices
            T1 = nn.Parameter(torch.eye(4, device=device).unsqueeze(0).expand(num_T, -1, -1).clone())
        # Initialize T2
        if init_T2 is not None:
            if len(init_T2.shape) == 2:  # Single 4x4 matrix
                T2 = nn.Parameter(init_T2.unsqueeze(0).expand(num_T, -1, -1).clone())
            else:  # Already in (num_T, 4, 4) format
                T2 = nn.Parameter(torch.tensor(init_T2, dtype=torch.float32, device=device))
        else:
            # Default initialization to num_T identity matrices
            T2 = nn.Parameter(torch.eye(4, device=device).unsqueeze(0).expand(num_T, -1, -1).clone())
        
        return T1, T2

    def forward(self, x, w_id, T_id, depth_list=None):
        """
        x: (N, 3) or (N, 4) - Input point coordinates
        w_id: (N,) - Point indices for selecting weights from base_weights
        T_id: (N,) - Indices for selecting transformation matrices
        """
        # Get base weights (one weight per point)
        base_weights = torch.sigmoid(self.weight)  # (num_points, 1)
        # Convert to homogeneous coordinates
        if x.shape[-1] == 3:
            ones = torch.ones(x.shape[0], 1, device=x.device)
            x_homogeneous = torch.cat([x, ones], dim=-1)  # (N, 4)
        else:
            x_homogeneous = x
        # Select corresponding transformation matrices for each point
        # self.T1 and self.T2 shape is (num_T, 4, 4)
        T1_batch = self.T1[T_id]  # (N, 4, 4)
        T2_batch = self.T2[T_id]  # (N, 4, 4)
        # Select corresponding weights for each point
        w_selected = base_weights[w_id.flatten()]  # (N, 1)
        # Apply transformations (batch matrix multiplication)
        x_homogeneous_expanded = x_homogeneous.unsqueeze(-1)  # (N, 4, 1)
        # Apply T1 and T2 transformations
        transformed_T1 = torch.bmm(T1_batch, x_homogeneous_expanded).squeeze(-1)  # (N, 4)
        transformed_T2 = torch.bmm(T2_batch, x_homogeneous_expanded).squeeze(-1)  # (N, 4)
        # Combine transformation results
        transformed = (1 - w_selected) * transformed_T1 + w_selected * transformed_T2
        
        if depth_list is not None:
            # Project 3D points to 2D image coordinates
            proj_points_homo = (self.cam_K @ transformed[:, :3].T).T  # (N, 3)
            proj_points = proj_points_homo[:, :2] / proj_points_homo[:, 2:3]  # (N, 2)
            
            # Convert to integer pixel coordinates for indexing
            proj_points_int = torch.round(proj_points).long()  # (N, 2)
            
            # Clamp coordinates to valid image bounds (assuming standard image size)
            # You might need to adjust these bounds based on your actual image dimensions
            H, W = depth_list.shape[-2:]  # Get height and width from depth maps
            proj_points_int[:, 0] = torch.clamp(proj_points_int[:, 0], 0, W-1)  # x coordinates
            proj_points_int[:, 1] = torch.clamp(proj_points_int[:, 1], 0, H-1)  # y coordinates
            
            # Sample depth values at projected locations
            # depth_list shape should be (num_T, H, W)
            batch_indices = T_id.flatten()  # (N,)
            y_coords = proj_points_int[:, 1]  # (N,)
            x_coords = proj_points_int[:, 0]  # (N,)
            point_depths = depth_list[batch_indices, y_coords, x_coords]  # (N,)
            
            # Convert back to 3D points using the sampled depth values
            # Unproject 2D pixel coordinates back to 3D using sampled depths
            cam_K_inv = torch.inverse(self.cam_K)  # (3, 3)
            
            # Create homogeneous pixel coordinates with sampled depths
            pixel_coords_homo = torch.stack([
                proj_points[:, 0] * point_depths,  # x * depth
                proj_points[:, 1] * point_depths,  # y * depth  
                point_depths                       # depth
            ], dim=1)  # (N, 3)
            
            # Unproject to 3D camera coordinates
            points_3d_reprojected = (cam_K_inv @ pixel_coords_homo.T).T  # (N, 3)
            
            return transformed[:, :3], points_3d_reprojected  # Return both transformed and reprojected points
        else:
            return transformed[:, :3], None  # Return transformed points and None for reprojected points

    # --- parameterization-neutral accessors (legacy free 4x4 matrices) ---------------------------
    def transform_parameters(self):
        return [self.T1, self.T2]

    def set_transforms_trainable(self, trainable):
        self.T1.requires_grad = trainable
        self.T2.requires_grad = trainable

    def assign_transform(self, index, T1, T2):
        """Overwrite transform ``index`` (only used when ``transform_reinit_apply`` is enabled)."""
        with torch.no_grad():
            self.T1.data[index] = T1.to(device=self.T1.device, dtype=self.T1.dtype)
            self.T2.data[index] = T2.to(device=self.T2.device, dtype=self.T2.dtype)

    def forward_components(self, x, w_id, T_id):
        """
        Return per-transform predictions and selected weights in addition to blended prediction.
        Args:
            x: (N,3) input points
            w_id: (N,) indices to pick point weights
            T_id: (N,) indices to pick transform per sample
        Returns:
            pred_T1: (N,3)
            pred_T2: (N,3)
            w_selected: (N,) selected weights in [0,1]
            blended: (N,3) (1-w) * T1(x) + w * T2(x)
        """
        base_weights = torch.sigmoid(self.weight)  # (num_points,1)
        if x.shape[-1] == 3:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            x_homogeneous = torch.cat([x, ones], dim=-1)
        else:
            x_homogeneous = x
        T1_batch = self.T1[T_id]
        T2_batch = self.T2[T_id]
        w_selected = base_weights[w_id.flatten()].squeeze(1)  # (N,)
        xh = x_homogeneous.unsqueeze(-1)
        transformed_T1 = torch.bmm(T1_batch, xh).squeeze(-1)[..., :3]
        transformed_T2 = torch.bmm(T2_batch, xh).squeeze(-1)[..., :3]
        blended = (1 - w_selected.unsqueeze(1)) * transformed_T1 + w_selected.unsqueeze(1) * transformed_T2
        return transformed_T1, transformed_T2, w_selected, blended


def _hat(v):
    """(..., 3) -> (..., 3, 3) skew-symmetric matrices."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def se3_exp(xi):
    """Exp: se(3) -> SE(3).  ``xi`` (..., 6) = (rho, phi): translation part, rotation vector.

    Computed as the matrix exponential of the 4x4 twist, which is exact, differentiable and
    well defined at ``phi = 0``; the rotation block is in SO(3) up to floating-point rounding.
    """
    twist = torch.zeros(xi.shape[:-1] + (4, 4), dtype=xi.dtype, device=xi.device)
    twist[..., :3, :3] = _hat(xi[..., 3:])
    twist[..., :3, 3] = xi[..., :3]
    return torch.linalg.matrix_exp(twist)


def project_to_se3(T):
    """Nearest rigid transform (rotation by SVD, det +1) for initialisations given as 4x4 matrices."""
    T = torch.as_tensor(T)
    U, _, Vh = torch.linalg.svd(T[..., :3, :3])
    d = torch.sign(torch.linalg.det(U @ Vh))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    out = torch.zeros_like(T)
    out[..., :3, :3] = U @ D @ Vh
    out[..., :3, 3] = T[..., :3, 3]
    out[..., 3, 3] = 1.0
    return out


class RigidTrans(Trans):
    """Two-part model whose transforms are rigid during optimisation (``transform_parameterization: se3``).

    T_k = C Exp(xi_k) C^-1 T_init,k  with  C = translation(pivot),  xi_k in R^6.

    ``T_init`` is a buffer (the existing rigid initialisation, unchanged); only ``xi`` is optimised,
    so every iterate is in SE(3).  The pivot (centroid of the window's source points) only
    conditions the rotation coordinates; it does not restrict the motion.  ``T1``/``T2`` are
    read-only properties with the legacy (num_T, 4, 4) layout so losses, priors and exports are
    unchanged.
    """

    def __init__(self, num_points, num_T, init_weights=None, init_T1=None, init_T2=None, cam_K=None,
                 device="cuda", pivot=None):
        nn.Module.__init__(self)
        self.weight = self._initialize_weights(num_points, init_weights, device)
        T1, T2 = Trans._initialize_transform_matrices(self, num_T, init_T1, init_T2, device)
        self.register_buffer("T1_init", project_to_se3(T1.detach().to(device=device, dtype=torch.float32)))
        self.register_buffer("T2_init", project_to_se3(T2.detach().to(device=device, dtype=torch.float32)))
        self.xi1 = nn.Parameter(torch.zeros(num_T, 6, device=device))
        self.xi2 = nn.Parameter(torch.zeros(num_T, 6, device=device))
        pivot = torch.zeros(3) if pivot is None else torch.as_tensor(pivot).reshape(3)
        self.register_buffer("pivot", pivot.to(device=device, dtype=torch.float32))
        self.cam_K = torch.tensor(cam_K, dtype=torch.float32, device=device)

    def _compose(self, xi, T_init):
        E = se3_exp(xi)
        C = torch.eye(4, device=xi.device, dtype=xi.dtype).expand_as(E).clone()
        C[..., :3, 3] = self.pivot
        C_inv = C.clone()
        C_inv[..., :3, 3] = -self.pivot
        return C @ E @ C_inv @ T_init

    @property
    def T1(self):
        return self._compose(self.xi1, self.T1_init)

    @property
    def T2(self):
        return self._compose(self.xi2, self.T2_init)

    def transform_parameters(self):
        return [self.xi1, self.xi2]

    def set_transforms_trainable(self, trainable):
        self.xi1.requires_grad = trainable
        self.xi2.requires_grad = trainable

    def assign_transform(self, index, T1, T2):
        with torch.no_grad():
            self.T1_init[index] = project_to_se3(T1.to(device=self.T1_init.device, dtype=torch.float32))
            self.T2_init[index] = project_to_se3(T2.to(device=self.T2_init.device, dtype=torch.float32))
            self.xi1.data[index] = 0.0
            self.xi2.data[index] = 0.0


def make_trans(parameterization="matrix", pivot=None, **kwargs):
    """``matrix``: legacy free 4x4 matrices (default, unchanged).  ``se3``: rigid ``RigidTrans``."""
    if parameterization in (None, "matrix"):
        return Trans(**kwargs)
    if parameterization == "se3":
        return RigidTrans(pivot=pivot, **kwargs)
    raise ValueError(f"Unknown transform_parameterization: {parameterization!r} (expected 'matrix' or 'se3')")


class Optimizer:
    def __init__(self, model, weight_lr, transform_lr, args=None, max_epoch=1000):

        self.max_epoch = max_epoch
        self.optimizer = torch.optim.Adam([
            {'params': [model.weight], 'lr': weight_lr},
            {'params': model.transform_parameters(), 'lr': transform_lr}
        ])
        
        use_scheduler = getattr(args, 'use_scheduler', True)
        scheduler_type = getattr(args, 'scheduler_type', 'cosine')
        
        if use_scheduler:
            if scheduler_type == 'cosine_restart':
                # CosineAnnealingWarmRestarts - recommended option.
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, 
                    T_0=100,           # First restart period.
                    T_mult=2,          # Period multiplier after each restart.
                    eta_min=1e-6,      # Minimum learning rate.
                    last_epoch=-1
                )
                self.scheduler_type = 'cosine_restart'
                
            elif scheduler_type == 'cosine':
                # CosineAnnealingLR - simple cosine schedule.
                max_epochs = self.max_epoch
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max_epochs,
                    eta_min=1e-6
                )
                self.scheduler_type = 'cosine'
                
            elif scheduler_type == 'plateau':
                # ReduceLROnPlateau - loss-based schedule.
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5,        # More aggressive decay.
                    patience=50,       # Shorter patience.
                    verbose=True, 
                    min_lr=1e-6,
                    threshold=1e-4     # Loss improvement threshold.
                )
                self.scheduler_type = 'plateau'
                
            elif scheduler_type == 'exponential':
                # ExponentialLR - exponential decay.
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=0.995        # Per-step decay factor.
                )
                self.scheduler_type = 'exponential'
                
            elif scheduler_type == 'step':
                # StepLR - step decay.
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=200,     # Decay every 200 steps.
                    gamma=0.8          # Decay factor.
                )
                self.scheduler_type = 'step'
            
            # Early stopping state.
            self.best_loss = float('inf')
            self.patience_counter = 0
            self.patience = getattr(args, 'early_stop_patience', 300)
    
    def step(self):
        """Perform optimization step"""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.optimizer.zero_grad()

    def scheduler_step(self, loss):
        """Step the learning rate scheduler"""
        if hasattr(self, 'scheduler'):
            if self.scheduler_type == 'plateau':
                # ReduceLROnPlateau needs the loss value.
                if loss is not None:
                    self.scheduler.step(loss)
            elif self.scheduler_type in ['cosine_restart', 'cosine', 'exponential', 'step']:
                # Other schedulers step by epoch.
                self.scheduler.step()
            
            # Early stopping check.
            if loss is not None:
                if loss < self.best_loss - 1e-6:  # Add a small improvement threshold.
                    self.best_loss = loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    # print(f"Early stopping triggered after {self.patience} epochs without improvement")
                    return True
        
        return False


class TransformLossComputer:
    """
    Unified loss computer for transformation optimization.

    It encapsulates:
    - correspondence main loss
    - entropy regularization
    - smoothness regularization
    - projection matching loss
    - init-weight BCE regularization
    - weighted total loss
    """

    def __init__(
        self,
        *,
        loss_type,
        cam_k_tensor,
        w_p,
        w_m,
        w_e,
        w_s,
        w_init,
        weight_layout="flat",
        allow_normalized_l1=True,
        invalid_loss_mode="raise",
    ):
        self.loss_type = loss_type
        self.cam_k_tensor = cam_k_tensor
        self.w_p = w_p
        self.w_m = w_m
        self.w_e = w_e
        self.w_s = w_s
        self.w_init = w_init
        self.weight_layout = weight_layout
        self.allow_normalized_l1 = allow_normalized_l1
        self.invalid_loss_mode = invalid_loss_mode
        self.consistency_loss = ConsistencyLoss()

    @staticmethod
    def _compute_normalized_error(pred_points, target_points, source_points, epsilon=0.001):
        abs_error = torch.norm(pred_points - target_points, dim=1)
        motion_magnitude = torch.norm(target_points - source_points, dim=1)
        return abs_error / (motion_magnitude + epsilon)

    def _compute_main_loss(self, pred_t1, pred_t2, w_sel, blended, target_points, source_points):
        loss_type = self.loss_type
        if loss_type == "l1":
            res1 = torch.abs(pred_t1 - target_points).mean(dim=1)
            res2 = torch.abs(pred_t2 - target_points).mean(dim=1)
            return ((1 - w_sel) * res1 + w_sel * res2).mean()

        if loss_type == "huber":
            res1 = F.huber_loss(pred_t1, target_points, reduction="none").mean(dim=1)
            res2 = F.huber_loss(pred_t2, target_points, reduction="none").mean(dim=1)
            return ((1 - w_sel) * res1 + w_sel * res2).mean()

        if loss_type == "normalized_l1":
            if not self.allow_normalized_l1:
                raise ValueError(f"Invalid loss type: {loss_type}")
            res1 = self._compute_normalized_error(pred_t1, target_points, source_points)
            res2 = self._compute_normalized_error(pred_t2, target_points, source_points)
            return ((1 - w_sel) * res1 + w_sel * res2).mean()

        if loss_type == "normalized_huber":
            norm_err1 = self._compute_normalized_error(pred_t1, target_points, source_points)
            norm_err2 = self._compute_normalized_error(pred_t2, target_points, source_points)
            res1 = F.huber_loss(
                norm_err1.unsqueeze(1),
                torch.zeros_like(norm_err1.unsqueeze(1)),
                reduction="none",
                delta=0.1,
            ).squeeze(1)
            res2 = F.huber_loss(
                norm_err2.unsqueeze(1),
                torch.zeros_like(norm_err2.unsqueeze(1)),
                reduction="none",
                delta=0.1,
            ).squeeze(1)
            return ((1 - w_sel) * res1 + w_sel * res2).mean()

        if loss_type == "chamfer":
            return chamfer_loss(blended, target_points)

        if self.invalid_loss_mode == "l1":
            res1 = torch.abs(pred_t1 - target_points).mean(dim=1)
            res2 = torch.abs(pred_t2 - target_points).mean(dim=1)
            return ((1 - w_sel) * res1 + w_sel * res2).mean()
        raise ValueError(f"Invalid loss type: {loss_type}")

    def _compute_weight_regs(self, weight_logits, knn_idx, neighbor_w):
        if self.weight_layout == "flat":
            current_weights = torch.sigmoid(weight_logits).squeeze(1)
            entropy_input = current_weights.unsqueeze(1)
            smooth_input = current_weights.unsqueeze(1)
        elif self.weight_layout == "column":
            current_weights = torch.sigmoid(weight_logits)
            entropy_input = current_weights
            smooth_input = current_weights
        else:
            raise ValueError(f"Unknown weight layout: {self.weight_layout}")

        entropy_reg = entropy_loss(entropy_input)
        if knn_idx is not None:
            smooth_reg = self.consistency_loss.forward(
                smooth_input,
                knn_idx,
                smooth_input,
                neighbor_weights=neighbor_w,
            )
        else:
            smooth_reg = torch.zeros(1, device=weight_logits.device)
        return entropy_reg, smooth_reg

    def compute_total_loss(
        self,
        *,
        pred_t1,
        pred_t2,
        w_sel,
        blended,
        target_points,
        source_points,
        weight_logits,
        init_weights_tensor,
        knn_idx=None,
        neighbor_w=None,
        valid_init_idx=None,
        extra_loss=None,
    ):
        """``extra_loss``: optional, already-weighted prior term (Phase B seed / Phase C contact).

        ``None`` leaves the baseline total unchanged; ``main_loss`` never includes it.
        """
        main_loss = self._compute_main_loss(
            pred_t1, pred_t2, w_sel, blended, target_points, source_points
        )

        entropy_reg, smooth_reg = self._compute_weight_regs(weight_logits, knn_idx, neighbor_w)

        proj1 = projection_matching_loss(
            pred_t1[..., :3], target_points[..., :3], self.cam_k_tensor, delta=1.0, reduction="none"
        )
        proj2 = projection_matching_loss(
            pred_t2[..., :3], target_points[..., :3], self.cam_k_tensor, delta=1.0, reduction="none"
        )
        projection_loss = ((1 - w_sel) * proj1 + w_sel * proj2).mean()

        logits = weight_logits.squeeze(1)
        if valid_init_idx is not None and valid_init_idx.numel() > 0:
            init_reg = F.binary_cross_entropy_with_logits(
                logits[valid_init_idx], init_weights_tensor[valid_init_idx]
            )
        else:
            init_reg = F.binary_cross_entropy_with_logits(logits, init_weights_tensor)

        total_loss = (
            self.w_p * projection_loss
            + self.w_m * main_loss
            + self.w_e * entropy_reg
            + self.w_s * smooth_reg
            + self.w_init * init_reg
        )
        if extra_loss is not None:
            total_loss = total_loss + extra_loss
        if _audit_trace_enabled():   # audit only: detached references, read by the solver's trace hook
            self._trace_terms = dict(main=main_loss.detach(), projection=projection_loss.detach(),
                                     entropy=entropy_reg.detach(), smooth=smooth_reg.detach(), init=init_reg.detach(),
                                     extra=None if extra_loss is None else extra_loss.detach(), total=total_loss.detach(),
                                     weights=dict(w_p=self.w_p, w_m=self.w_m, w_e=self.w_e, w_s=self.w_s, w_init=self.w_init))
        return total_loss, main_loss
            
