import torch
import os
import torch as th
import numpy as np
from .scheduler import get_schedule_jump
from .respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor
from utils import normalize_image, save_grid, save_image


class DDRMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(use_timesteps, conf, **kwargs)
        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = th.split(model_output, C, dim=1)
        return model_output

    @th.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir=None,
        **kwargs,
    ):
        # copied from ddrm(https://github.com/bahjat-kawar/ddrm)
        mask = model_kwargs["gt_keep_mask"]
        # missing_r = th.nonzero(mask == 0).long().reshape(-1) * 3
        missing_r = th.nonzero(mask[0][0].view(-1) == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = th.cat([missing_r, missing_g, missing_b], dim=0)
        if device is None:
            device = next(model.parameters()).device
        B, C, H, W = shape
        assert isinstance(shape, (tuple, list))
        if self.mode == "inpaint":
            H_funcs = Inpainting(
                C,
                H,
                missing,
                device=device,
            )
        else:
            H_funcs = SuperResolution(
                C, H, self.scale, device=device
            )
        gt = model_kwargs["gt"]
        y_0 = H_funcs.H(gt)
        sigma_0 = 0.0  # maybe 0
        y_0 = y_0 + sigma_0 * th.randn_like(y_0)

        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], C, H, W)
        pinv_y_0 += (
            H_funcs.H_pinv(H_funcs.H(th.ones_like(pinv_y_0))
                           ).reshape(*pinv_y_0.shape)
            - 1
        )

        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        if sample_dir is not None:
            print("making sample_dir, ", sample_dir)
            os.makedirs(sample_dir, exist_ok=True)

        times = get_schedule_jump(**conf["ddrm.schedule_jump_params"])
        time_pairs = list(zip(times[:-2], times[1:-1]))
        if progress:
            from tqdm.auto import tqdm

            time_pairs = tqdm(time_pairs)

        x = image_after_step

        # copied from ddrm/functions/denoising.py
        singulars = H_funcs.singulars()
        Sigma = th.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=x.device)
        Sigma[: singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]

        largest_alphas = _extract_into_tensor(
            self.alphas_cumprod,
            th.tensor(times[0], device=device),
            th.Size([1] * len(x.shape)),
        )
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = th.where(
            singulars * largest_sigmas[0, 0, 0, 0] > sigma_0
        )
        inv_singulars_and_zero = th.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(
            singulars.device
        )
        inv_singulars_and_zero[large_singulars_index] = (
            sigma_0 / singulars[large_singulars_index]
        )

        init_y = th.zeros(x.shape[0], x.shape[1] *
                          x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[
            :, large_singulars_index[0]
        ] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1,
                                          1) ** 2 - inv_singulars_and_zero**2
        remaining_s = (
            remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            .clamp_min(0.0)
            .sqrt()
        )
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        xt = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)

        # ! not sure what to set
        etaB = 1.0
        etaC = 0.0
        etaA = 0.0

        for t_last, t_cur in time_pairs:
            t_last_t = th.tensor([t_last] * shape[0], device=device)
            t_cur_t = th.tensor([t_cur] * shape[0], device=device)
            at = _extract_into_tensor(self.alphas_cumprod, t_last_t, xt.shape)
            at_next = _extract_into_tensor(
                self.alphas_cumprod, t_cur_t, xt.shape)

            if cond_fn is not None:
                model_fn = self._wrap_model(model)
                B, C = xt.shape[:2]
                assert t_last_t.shape == (B,)
                model_output = model_fn(
                    xt, self._scale_timesteps(t_last_t), **model_kwargs
                )
                assert model_output.shape == (B, C * 2, *xt.shape[2:])
                _, model_var_values = th.split(model_output, C, dim=1)
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t_last_t, xt.shape
                )
                max_log = _extract_into_tensor(
                    np.log(self.betas), t_last_t, xt.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
                with th.enable_grad():
                    gradient = cond_fn(
                        xt, self._scale_timesteps(t_last_t), **model_kwargs
                    )
                    xt = xt + model_variance * gradient

            et = self._get_et(model, xt, t_last_t, model_kwargs)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0,
                                              0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, : U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

            falses = th.zeros(
                V_t_x0.shape[1] - singulars.shape[0], dtype=th.bool, device=xt.device
            )
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = th.hstack((cond_before_lite, falses))
            cond_after = th.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = th.sqrt(sigma_next**2 - std_nextC**2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = th.sqrt(sigma_next**2 - std_nextA**2)

            diff_sigma_t_nextB = th.sqrt(
                sigma_next**2
                - sigma_0**2 / singulars[cond_before_lite] ** 2 * (etaB**2)
            )

            Vt_xt_mod_next = (
                V_t_x0
                + sigma_tilde_nextC * H_funcs.Vt(et)
                + std_nextC * th.randn_like(V_t_x0)
            )

            # less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = (
                V_t_x0[:, cond_after]
                + sigma_tilde_nextA *
                ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
                + std_nextA * th.randn_like(V_t_x0[:, cond_after])
            )

            # noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = (
                Sig_inv_U_t_y[:, cond_before_lite] * etaB
                + (1 - etaB) * V_t_x0[:, cond_before]
                + diff_sigma_t_nextB * th.randn_like(U_t_y)[:, cond_before_lite]
            )

            # aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)
            xt = xt_next

            if sample_dir is not None:
                save_grid(
                    normalize_image(x0_t.clamp(-1, 1)),
                    os.path.join(sample_dir, f"pred-{t_last}.png"),
                )

        xt = xt.clamp(-1, 1)
        return {
            "sample": xt,
            "gt": model_kwargs["gt"],
        }


# Utils


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, : singulars.shape[0]] = temp[:,
                                             : singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(
            v.shape[0], M.shape[0]
        )

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, : self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(
            channels * img_dim**2 - missing_indices.shape[0]
        ).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = (
            torch.Tensor(
                [i for i in range(channels * img_dim**2)
                 if i not in missing_indices]
            )
            .to(device)
            .long()
        )

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, : self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return (
            out.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

    def Vt(self, vec):
        temp = (
            vec.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out = torch.zeros_like(temp)
        out[:, : self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(
            H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels *
                                   self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)                                           ::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)
                               ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # repatch the patches into an image
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
                               ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(
            vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :,
                                                           :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:,
                                                                                   :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
