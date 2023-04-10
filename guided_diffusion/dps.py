from torchvision import torch
from torch.nn import functional as F
import yaml
from functools import partial
import torch
from abc import ABC, abstractmethod
import os
import torch as th
import numpy as np
from .scheduler import get_schedule_jump
from .respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor
import torch.nn as nn
from utils import normalize_image, save_grid, save_image


class DPSSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(use_timesteps, conf, **kwargs)
        self.step_size = conf.get("dps.step_size", 0.5)  # 1 for CelebaAHQ
        self.eta = conf.get("dps.eta", 1.0)  # dps uses ddpm
        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)

    def p_sample(
        self,
        model,
        x,
        t,
        eta=0.0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        sample_dir=None,
        **kwargs,
    ):
        # condition mean
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

        eta = self.eta
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = self.predict_eps_from_x_start(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(
            self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )

        sample = mean_pred
        if t[0] != 0:
            sample += sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (coef1 * x_t - pred_xstart) / coef2

    def q_sample(self, x_start, t):
        noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        coef1 = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        coef2 = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return coef1 * x_start + coef2 * noise

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

        gt = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]

        if self.mode == "inpaint":
            self.operator = get_operator("inpainting", device=device)
        elif self.mode == "super_resolution":
            self.operator = get_operator("super_resolution", device=device,
                                         in_shape=gt.shape[-2:],
                                         scale_factor=self.scale)
        else:
            raise ValueError(f"Unkown mode: {self.mode}")

        self.noiser = get_noise(
            conf.get("dps.noise_type", "gaussian"), sigma=0.0001
        )  # zero-noise gaussian
        self.cond_method = get_conditioning_method(
            "ps", self.operator, self.noiser, scale=self.step_size
        )
        self.measurement_cond_fn = self.cond_method.conditioning
        # construct measurement
        measurement = self.noiser(
            self.operator.forward(gt, mask=mask))  # masked x0
        measurement_cond_fn = partial(self.measurement_cond_fn, mask=mask)

        if sample_dir is not None:
            print("making sample_dir, ", sample_dir)
            os.makedirs(sample_dir, exist_ok=True)

        times = get_schedule_jump(**conf["dps.schedule_jump_params"])
        time_pairs = list(zip(times[:-2], times[1:-1]))
        if progress:
            from tqdm.auto import tqdm

            time_pairs = tqdm(time_pairs)

        xt = image_after_step
        for t_last, t_cur in time_pairs:
            t_last_t = th.tensor([t_last] * shape[0], device=device)
            with th.enable_grad():
                xt = xt.requires_grad_()
                out = self.p_sample(x=xt, t=t_last_t, model=model,
                                    cond_fn=cond_fn, model_kwargs=model_kwargs)

                if sample_dir is not None:
                    save_grid(
                        normalize_image(out['pred_xstart'].clamp(-1, 1)),
                        os.path.join(
                            sample_dir, f"pred-{t_cur}.png"
                        )
                    )

                noisy_measurement = self.q_sample(measurement, t=t_last_t)

                xt, distance = measurement_cond_fn(
                    x_t=out["sample"],
                    measurement=measurement,
                    noisy_measurement=noisy_measurement,
                    x_prev=xt,
                    x_0_hat=out["pred_xstart"],
                )
                xt = xt.detach_()

                out["sample"] = xt
                out["gt"] = gt
                yield out

        th.cuda.empty_cache()


# DPS utils, copied from DPS repo
# condition functions

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == "gaussian":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name="vanilla")
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name="projection")
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name="mcg")
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(
        self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs
    ):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(
            data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name="ps")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        # x_t -= self.scale / norm_grad
        return x_t, norm


@register_conditioning_method(name="ps+")
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


# noise measurements
"""This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n."""


# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="noise")
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, device):
        self.device = device

    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get("mask", None).to(self.device)
        except:
            raise ValueError("Require mask")

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = nn.AdaptiveAvgPool2d(
            (in_shape[0] // scale_factor, in_shape[1] // scale_factor))

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


# =============
# Noise classes
# =============

__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
