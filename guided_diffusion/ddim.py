import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm

from utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma", 0.0)

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                 ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                 t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
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
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0)
        mask = model_kwargs["gt_keep_mask"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # replace surrounding
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                 f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                 f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


class R_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

    @staticmethod
    def resample(m, w, n):
        """
        m: max number of index
        w: un-normalized probability
        n: number of indices to be selected
        """
        if max([(math.isnan(i) or math.isinf(i)) for i in w]):
            w = np.ones_like(w)
        if w.sum() < 1e-6:
            w = np.ones_like(w)

        w = n * (w / w.sum())
        c = [int(i) for i in w]
        r = [i - int(i) for i in w]
        added_indices = []
        for i in range(m):
            for j in range(c[i]):
                added_indices.append(i)
        if len(added_indices) != n:
            R = n - sum(c)
            indices_r = torch.multinomial(torch.tensor(r), R)
            for i in indices_r:
                added_indices.append(i)
        logging_info(
            "Indices after Resampling: %s"
            % (" ".join(["%.d" % i for i in sorted(added_indices)]))
        )
        return added_indices

    @staticmethod
    def gaussian_pdf(x, mean, std=1):
        return (
            1
            / (math.sqrt(2 * torch.pi) * std)
            * torch.exp(-((x - mean) ** 2).sum() / (2 * std**2))
        )

    def resample_based_on_x_prev(
        self,
        x_t,
        x_prev,
        x_pred_prev,
        mask,
        keep_n_samples=None,
        temperature=100,
        p_cal_method="mse_inverse",
        pred_x0=None,
    ):
        if p_cal_method == "mse_inverse":  # same intuition but empirically better
            mse = torch.tensor(
                [((x_prev * mask - i * mask) ** 2).sum() for i in x_pred_prev]
            )
            mse /= mse.mean()
            p = torch.softmax(temperature / mse, dim=-1)
        elif p_cal_method == "gaussian":
            p = torch.tensor(
                [self.gaussian_pdf(x_prev * mask, i * mask)
                 for i in x_pred_prev]
            )
        else:
            raise NotImplementedError
        resample_indices = self.resample(
            x_t.shape[0], p, x_t.shape[0] if keep_n_samples is None else keep_n_samples
        )
        x_t = torch.stack([x_t[i] for i in resample_indices], dim=0)
        x_pred_prev = torch.stack([x_pred_prev[i]
                                  for i in resample_indices], dim=0)
        pred_x0 = (
            torch.stack([pred_x0[i] for i in resample_indices], dim=0)
            if pred_x0 is not None
            else None
        )
        logging_info(
            "Resampling with probability %s" % (
                " ".join(["%.3lf" % i for i in p]))
        )
        return x_t, x_pred_prev, pred_x0

    def p_sample_loop(
        self,
        model_fn,
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
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]
        # x_forwards = self.x_forward_sample(x0, "from_middle")
        x_forwards = self.x_forward_sample(x0, "from_0")
        x_t = img

        for cur_t, prev_t in tqdm(time_pairs):
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            x_prev = x_forwards[prev_t]
            output = self.p_sample(
                model_fn,
                x=x_t,
                t=torch.tensor([cur_t] * shape[0], device=device),
                prev_t=torch.tensor([prev_t] * shape[0], device=device),
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )

            x_pred_prev, x_pred_x0 = output["x_prev"], output["pred_x0"]
            x_t, x_pred_prev, pred_x0 = self.resample_based_on_x_prev(
                x_t=x_t,
                x_prev=x_prev,
                x_pred_prev=x_pred_prev,
                mask=mask,
                pred_x0=x_pred_x0,
            )
            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles", f"mid-{prev_t}.png"),
                )
                save_grid(
                    normalize_image(pred_x0),
                    os.path.join(sample_dir, "middles", f"pred-{prev_t}.png"),
                )

        x_t = self.resample_based_on_x_prev(
            x_t=x_t,
            x_prev=x0,
            x_pred_prev=x_t,
            mask=mask,
            keep_n_samples=conf["resample.keep_n_samples"],
        )[0]

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


# implemenet
class O_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

        assert conf.get("optimize_xt.optimize_xt",
                        False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_optimize_xt = conf.get(
            "optimize_xt.num_iteration_optimize_xt", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel", False):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])

        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        if self.mode == "inpaint":
            def loss_fn(_x0, _pred_x0, _mask):
                ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
                return ret
        elif self.mode == "super_resolution":
            size = x.shape[-1]
            downop = nn.AdaptiveAvgPool2d(
                (size // self.scale, size // self.scale))

            def loss_fn(_x0, _pred_x0, _mask):
                down_x0 = downop(_x0)
                down_pred_x0 = downop(_pred_x0)
                ret = torch.sum((down_x0 - down_pred_x0) ** 2)
                return ret
        else:
            raise ValueError("Unkown mode: {self.mode}")

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)

        def get_update(
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                     *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]

        # condition mean
        if cond_fn is not None:
            model_fn = self._wrap_model(model_fn)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            with torch.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)
        # optimize
        with torch.enable_grad():
            origin_x = x.clone().detach()
            x = x.detach().requires_grad_()
            e_t = get_et(_x=x, _t=t)
            pred_x0 = get_predx0(
                _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
            )
            prev_loss = loss_fn(x0, pred_x0, mask).item()

            logging_info(f"step: {t[0].item()} lr_xt {lr_xt:.8f}")
            for step in range(self.num_iteration_optimize_xt):
                loss = loss_fn(x0, pred_x0, mask) + \
                    coef_xt_reg * reg_fn(origin_x, x)
                x_grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0].detach()
                new_x = x - lr_xt * x_grad

                logging_info(
                    f"grad norm: {torch.norm(x_grad, p=2).item():.3f} "
                    f"{torch.norm(x_grad * mask, p=2).item():.3f} "
                    f"{torch.norm(x_grad * (1. - mask), p=2).item():.3f}"
                )

                while self.use_adaptive_lr_xt and True:
                    with torch.no_grad():
                        e_t = get_et(new_x, _t=t)
                        pred_x0 = get_predx0(
                            new_x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                        )
                        new_loss = loss_fn(x0, pred_x0, mask) + coef_xt_reg * reg_fn(
                            origin_x, new_x
                        )
                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.8
                            logging_info(
                                "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                                % (loss.item(), new_loss.item(), lr_xt)
                            )
                            del new_x, e_t, pred_x0, new_loss
                            new_x = x - lr_xt * x_grad

                x = new_x.detach().requires_grad_()
                e_t = get_et(x, _t=t)
                pred_x0 = get_predx0(
                    x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                del loss, x_grad
                torch.cuda.empty_cache()

        # after optimize
        with torch.no_grad():
            new_loss = loss_fn(x0, pred_x0, mask).item()
            logging_info("Loss Change: %.3lf -> %.3lf" % (prev_loss, new_loss))
            new_reg = reg_fn(origin_x, new_x).item()
            logging_info("Regularization Change: %.3lf -> %.3lf" % (0, new_reg))
            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            del origin_x, prev_loss
            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )

        return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_loss}

    def p_sample_loop(
        self,
        model_fn,
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
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            assert not conf["optimize_xt.filter_xT"]
            img = noise
        else:
            xT_shape = (
                shape
                if not conf["optimize_xt.filter_xT"]
                else tuple([20] + list(shape[1:]))
            )
            img = torch.randn(xT_shape, device=device)

        if conf["optimize_xt.filter_xT"]:
            xT_losses = []
            for img_i in img:
                xT_losses.append(
                    self.p_sample(
                        model_fn,
                        x=img_i.unsqueeze(0),
                        t=torch.tensor([self.steps[0]] * 1, device=device),
                        prev_t=torch.tensor([0] * 1, device=device),
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        lr_xt=self.lr_xt,
                        coef_xt_reg=self.coef_xt_reg,
                    )["loss"]
                )
            img = img[torch.argsort(torch.tensor(xT_losses))[: shape[0]]]

        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        x_t = img
        # set up hyper paramer for this run
        lr_xt = self.lr_xt
        coef_xt_reg = self.coef_xt_reg
        loss = None

        status = None
        for cur_t, prev_t in tqdm(time_pairs):
            if cur_t > prev_t:  # denoise
                status = "reverse"
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                output = self.p_sample(
                    model_fn,
                    x=x_t,
                    t=cur_t,
                    prev_t=prev_t,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                )
                x_t = output["x_prev"]
                loss = output["loss"]

                # lr decay
                if self.lr_xt_decay != 1.0:
                    logging_info(
                        "Learning rate of xt decay: %.5lf -> %.5lf."
                        % (lr_xt, lr_xt * self.lr_xt_decay)
                    )
                lr_xt *= self.lr_xt_decay
                if self.coef_xt_reg_decay != 1.0:
                    logging_info(
                        "Coefficient of regularization decay: %.5lf -> %.5lf."
                        % (coef_xt_reg, coef_xt_reg * self.coef_xt_reg_decay)
                    )
                coef_xt_reg *= self.coef_xt_reg_decay

                if conf["debug"]:
                    from utils import normalize_image, save_grid

                    os.makedirs(os.path.join(
                        sample_dir, "middles"), exist_ok=True)
                    save_grid(
                        normalize_image(x_t),
                        os.path.join(
                            sample_dir, "middles", f"mid-{prev_t[0].item()}.png"
                        ),
                    )
                    save_grid(
                        normalize_image(output["pred_x0"]),
                        os.path.join(
                            sample_dir, "middles", f"pred-{prev_t[0].item()}.png"
                        ),
                    )
            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                    )
                status = "forward"
                assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                with torch.no_grad():
                    x_t = self._undo(x_t, prev_t)
                # undo lr decay
                logging_info(f"Undo step: {cur_t}")
                lr_xt /= self.lr_xt_decay
                coef_xt_reg /= self.coef_xt_reg_decay

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}

    def get_updated_xt(self, model_fn, x, t, model_kwargs, lr_xt, coef_xt_reg):
        return self.p_sample(
            model_fn,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
        )["x"]
