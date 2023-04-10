import os
import tqdm
import torch as th
import numpy as np
from collections import defaultdict
from .gaussian_diffusion import _extract_into_tensor
from .respace import SpacedDiffusion
from .scheduler import get_schedule_jump
from utils import normalize_image, save_image, save_grid


class DDNMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(use_timesteps, conf, **kwargs)
        self.sigma_y = conf.get("ddnm.sigma_y", 0.0)
        self.eta = conf.get("ddnm.eta", 0.85)
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

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        pred_xstart=None,
        idx_wall=-1,
        sample_dir=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if cond_fn is not None:
            model_fn = self._wrap_model(model)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
            with th.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        with th.no_grad():

            A = kwargs.get("A")
            Ap = kwargs.get("Ap")
            x0 = model_kwargs["gt"]
            y = A(x0)
            mask = model_kwargs["gt_keep_mask"]

            def process_xstart(x):
                if denoised_fn is not None:
                    x = denoised_fn(x)
                if clip_denoised:
                    return x.clamp(-1, 1)
                return x

            e_t = self._get_et(model, x, t, model_kwargs)
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            prev_t = t - 1
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigma_t = (1 - alpha_t**2).sqrt()

            pred_x0 = process_xstart(
                (x - e_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt())

            single_sigma_t = sigma_t[0][0][0][0]
            single_alpha_t = alpha_prev[0][0][0][0]
            if single_sigma_t >= single_alpha_t * self.sigma_y:
                lambda_t = 1.0
                gamma_t = (sigma_t**2 - (alpha_prev * self.sigma_y) ** 2).sqrt()
            else:
                lambda_t = (sigma_t) / (alpha_prev * self.sigma_y)
                gamma_t = 0.0

            # DDNM modification
            pred_x0 = pred_x0 - lambda_t * Ap(A(pred_x0) - y)

            eta = self.eta
            c1 = (1 - alpha_prev).sqrt() * eta
            c2 = (1 - alpha_prev).sqrt() * ((1 - eta**2) ** 0.5)

            x_prev = alpha_prev.sqrt() * pred_x0 + gamma_t * (
                c1 * th.randn_like(pred_x0) + c2 * e_t
            )

            result = {
                "sample": x_prev,
                "pred_xstart": pred_x0,
                "gt": model_kwargs["gt"],
            }
            return result

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None,
        sample_dir=None,
        **kwargs,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        # inpainting
        mask = model_kwargs["gt_keep_mask"]
        if self.mode == "inpaint":
            def A(z): return z * mask
            Ap = A
        elif self.mode == "super_resolution":
            def PatchUpsample(x, scale):
                n, c, h, w = x.shape
                x = th.zeros(n, c, h, scale, w, scale).to(x.device) + \
                    x.view(n, c, h, 1, w, 1)
                return x.view(n, c, scale*h, scale*w)

            size = shape[-1]
            A = th.nn.AdaptiveAvgPool2d((size // self.scale, size//self.scale))
            def Ap(z): return PatchUpsample(z, self.scale)
        else:
            raise ValueError("Unkown mode")

        self.gt_noises = None
        pred_xstart = None
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        if sample_dir is not None:
            os.makedirs(sample_dir, exist_ok=True)

        if conf["ddnm.schedule_jump_params"]:
            times = get_schedule_jump(**conf["ddnm.schedule_jump_params"])
            time_pairs = list(zip(times[:-2], times[1:-1]))
            if progress:
                from tqdm.auto import tqdm

                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                t_last_t = th.tensor([t_last] * shape[0], device=device)
                if t_cur < t_last:
                    # denoise
                    with th.no_grad():
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(
                            model,
                            image_after_step,
                            t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart,
                            A=A,
                            Ap=Ap,
                        )
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]

                        if sample_dir is not None:
                            save_grid(
                                normalize_image(pred_xstart.clamp(-1, 1)),
                                os.path.join(sample_dir, f"pred-{t_cur}.png"),
                            )
                        yield out
                else:
                    # ad dnoise
                    t_shift = conf.get("inpa_inj_time_shift", 1)

                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(
                        image_before_step,
                        image_after_step,
                        est_x_0=out["pred_xstart"],
                        t=t_last_t + t_shift,
                        debug=False,
                    )
                    pred_xstart = out["pred_xstart"]
