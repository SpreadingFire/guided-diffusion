"""
这段代码最初是作为Ho等人的扩散模型的PyTorch移植版本：
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

已经添加了Docstrings（文档字符串），DDIM采样和新的beta调度集合。

这段代码的目的是实现一个基于扩散模型的图像生成和训练框架。它主要用于生成和训练扩散模型（diffusion models），
这是由Jonathan Ho等人提出的一种图像生成技术。代码提供了训练和采样的主要功能，并实现了不同的beta调度表、损失类型和变量类型等。
"""

import enum
import math
import numpy as np
import torch as th
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    获取预定义的beta调度表。

    该库中的beta调度表在num_diffusion_timesteps的极限情况下保持相似。
    可以添加新的beta调度表，但一旦提交，为了保持向后兼容性，不应该删除或更改。
    """
    if schedule_name == "linear":
        # Ho等人的线性调度，扩展为适用于任意扩散步数。
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        # 余弦调度
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"未知的beta调度：{schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个beta调度，该调度离散化给定的alpha_t_bar函数，
    该函数定义了从t=[0,1]的（1-beta）的累积乘积。

    :param num_diffusion_timesteps: 要生成的beta数量。
    :param alpha_bar: 一个lambda函数，接受一个从0到1的参数t，并生成到该扩散过程部分的（1-beta）的累积乘积。
    :param max_beta: 使用的最大beta值；使用小于1的值以避免奇异性。
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    模型预测的输出类型。
    """
    PREVIOUS_X = enum.auto()  # 模型预测 x_{t-1}
    START_X = enum.auto()  # 模型预测 x_0
    EPSILON = enum.auto()  # 模型预测epsilon

class ModelVarType(enum.Enum):
    """
    用作模型输出方差的内容。

    添加LEARNED_RANGE选项以允许模型预测介于FIXED_SMALL和FIXED_LARGE之间的值，从而使其工作更简单。
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    """
    损失类型。
    """
    MSE = enum.auto()  # 使用原始MSE损失（以及学习方差时的KL）
    RESCALED_MSE = (
        enum.auto()
    )  # 使用原始MSE损失（在学习方差时使用RESCALED_KL）
    KL = enum.auto()  # 使用变分下界
    RESCALED_KL = enum.auto()  # 类似KL，但重缩放以估计完整的VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class GaussianDiffusion:
    """
    用于训练和采样扩散模型的工具。

    直接移植自这里，并随着时间的推移进行进一步的实验和适应。
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: 每个扩散时间步的beta的一维numpy数组，从T到1。
    :param model_mean_type: 一个ModelMeanType，确定模型输出内容。
    :param model_var_type: 一个ModelVarType，确定如何输出方差。
    :param loss_type: 一个LossType，确定使用的损失函数。
    :param rescale_timesteps: 如果为True，则将浮点时间步数传递给模型，使其始终按原始论文中的方式缩放（0到1000）。
    """
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # 使用float64以确保计算精度。
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas必须是一维数组"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 计算diffusion q(x_t | x_{t-1})和其他
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算后验 q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log计算被截断，因为在扩散链的开头，后验方差为0。
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        获取分布 q(x_t | x_0)。

        :param x_start: 无噪声输入的[N x C x ...]张量。
        :param t: 扩散步数（减1）。这里0指的是一步。
        :return: 一个包含均值、方差、log方差的元组，形状为x_start的形状。
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        对数据进行给定次数的扩散步骤。

        换句话说，从 q(x_t | x_0)采样
                进行采样。

        :param x_start: 初始数据批次。
        :param t: 扩散步数（减1）。这里0指的是一步。
        :param noise: 如果指定，则是分离出来的正态噪声。
        :return: 一个添加噪声后的 x_start 版本。
        """
        if noise is None:
            noise = th.randn_like(x_start)  # 如果没有提供噪声，则生成随机噪声
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散后验的均值和方差：

            q(x_{t-1} | x_t, x_0)

        :param x_start: 无噪声输入的[N x C x ...]张量。
        :param x_t: 当前时间步的张量x_t。
        :param t: 扩散步数（减1）。这里0指的是一步。
        :return: 一个包含均值、方差、log方差的元组，形状为x_start的形状。
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        应用模型获取 p(x_{t-1} | x_t)，以及 x_0 的预测。

        :param model: 模型，接受信号和一批时间步数作为输入。
        :param x: 在时间步 t 的 [N x C x ...] 张量。
        :param t: 一个包含时间步的1-D张量。
        :param clip_denoised: 如果为True，将去噪信号裁剪到[-1, 1]。
        :param denoised_fn: 如果不为None，这是一个应用于 x_start 预测的函数。
        :param model_kwargs: 如果不为None，是传递给模型的额外关键字参数的字典。这可以用于条件生成（conditioning）。
        :return: 一个包含以下键的字典：
                 - 'mean': 模型均值输出。
                 - 'variance': 模型方差输出。
                 - 'log_variance': 'variance'的对数。
                 - 'pred_xstart': 对 x_0 的预测。
        """
        if model_kwargs == None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # model_var_values 的范围是 [-1, 1]
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)  # 将去噪信号裁剪到[-1, 1]
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        从epsilon预测x_start。
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        从x_{t-1}预测x_start。
        """
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        从x_start预测epsilon。
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        根据时间步进行缩放。
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        计算上一步骤的均值，给定一个函数cond_fn，
        它计算条件概率log的梯度。

        :param cond_fn: 计算条件概率log梯度的函数。
        :param p_mean_var: 均值和方差。
        :param x: 当前样本。
        :param t: 当前时间步骤。
        :param model_kwargs: 传递给模型的额外关键字参数。
        :return: 计算得到的新均值。
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
               """
        使用带cond_fn的条件得分函数计算p_mean_variance的输出。

        :param cond_fn: 计算条件概率log梯度的函数。
        :param p_mean_var: 均值和方差。
        :param x: 当前样本。
        :param t: 当前时间步骤。
        :param model_kwargs: 传递给模型的额外关键字参数。
        :return: 带有新的预测x_start的输出字典。
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        # 从x_start预测epsilon。
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        在给定的时间步从模型中采样x_{t-1}。

        :param model: 用于采样的模型。
        :param x: 在x_{t-1}的当前张量。
        :param t: 时间步，从0开始表示第一次扩散步骤。
        :param clip_denoised: 如果为True，将x_start预测裁剪到[-1, 1]。
        :param denoised_fn: 如果不为None，这是一个应用于x_start预测的函数。
        :param cond_fn: 如果不为None，这是一个类似于模型的梯度函数。
        :param model_kwargs: 如果不为None，是传递给模型的额外关键字参数的字典。这可以用于条件生成（conditioning）。
        :return: 包含随机采样和x_0预测的字典。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时没有噪声
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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
        progress=False,
    ):
        """
        从模型生成样本。

        :param model: 模型模块。
        :param shape: 样本的形状，(N, C, H, W)。
        :param noise: 如果指定，则是从编码器生成的噪声。应与‘shape’形状相同。
        :param clip_denoised: 如果为True，将x_start预测裁剪到[-1, 1]。
        :param denoised_fn: 如果不为None，这是一个应用于x_start预测的函数。
        :param cond_fn: 如果不为None，这是一个类似于模型的梯度函数。
        :param model_kwargs: 如果不为None，是传递给模型的额外关键字参数的字典。这可以用于条件生成（conditioning）。
        :param device: 如果指定，则用于创建样本的设备。如果未指定，则使用模型参数的设备。
        :param progress: 如果为True，显示一个tqdm进度条。
        :return: 一个不可微分的样本批次。
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

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
    ):
        """
        从模型生成样本，并从DDIM的每一个时间步生成中间样本。

        参数与p_sample_loop()相同。
        返回一个生成器，生成的每个字典是p_sample()的返回值。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 懒加载，以便我们不依赖tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用DDIM从模型中采样x_{t-1}。

        与p_sample()的用法相同。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 通常我们的模型输出epsilon，但我们重新推导它
        # 以防我们使用 x_start 或 x_prev 预测。
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # 等式12。
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时没有噪声
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用DDIM逆ODE从模型中采样x_{t+1}。
        """
        assert eta == 0.0, "逆ODE仅用于确定性路径"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # 通常我们的模型输出epsilon，但我们重新推导它
        # 以防我们使用 x_start 或 x_prev 预测。
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # 反向等式12。
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

               + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
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
        eta=0.0,
    ):
        """
        使用DDIM从模型生成样本。

        使用方法与p_sample_loop()相同。
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
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
        eta=0.0,
    ):
        """
        使用DDIM从模型生成样本，并在每一步生成中间样本。

        使用方法与p_sample_loop_progressive()相同。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 懒加载，以便我们不依赖tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        获取变分下界的一个项。

        生成的单位是比特（而不是nats），以便与其他论文比较。

        :return: 一个包含以下键的字典：
                 - 'output': 一个形状为[N]的NLL或KL张量。
                 - 'pred_xstart': x_0预测。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # 在第一个时间步返回解码器NLL，
        # 否则返回KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        计算单个时间步的训练损失。

        :param model: 用于评估损失的模型。
        :param x_start: 输入的 [N x C x ...] 张量。
        :param t: 一批时间步的索引。
        :param model_kwargs: 如果不为None，是传递给模型的额外关键字参数的字典。
        :param noise: 如果指定，则是要去除的特定高斯噪声。
        :return: 一个包含“loss”键的字典，其中包含形状为[N]的张量。某些均值或方差设置可能还有其他键。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # 使用变分界限学习方差，但不要让
                # 它影响我们的均值预测。
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # 除以1000以等效于初始实现。
                    # 没有1/1000的因子，VB项会影响MSE项。
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        获取变分下界的先验KL项，以比特/维为单位。

        这个项不能被优化，因为它仅依赖于编码器。

        :param x_start: 输入的 [N x C x ...] 张量。
        :return: 一批[N]个KL值（以比特为单位），每个批次元素一个。
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个变分下界，以比特/维为单位，以及其他相关数量。

        :param model: 用于评估损失的模型。
        :param x_start: 输入的 [N x C x ...] 张量。
        :param clip_denoised: 如果为True，将去噪样本裁剪。
        :param model_kwargs: 如果不为None，是传递给模型的额外关键字参数的字典。

        :return: 一个包含以下键的字典：
                 - total_bpd: 每个批次元素的总变分下界。
                 - prior_bpd: 变分下界中的先验项。
                 - vb: 一个 [N x T] 张量，表示变分下界中的项。
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                                  - mse: 一个 [N x T] 张量，表示均方误差损失项。
        """
        device = x_start.device

        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            # 提取时间步 t 的采样步骤
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            with th.no_grad():
                out = self._vb_terms_bpd(
                    model=model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])

            mse.append(mean_flat((noise - self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])) ** 2).detach())
            xstart_mse.append(mean_flat((x_start - out["pred_xstart"]) ** 2).detach())

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)
        prior_bpd = self._prior_bpd(x_start)

        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从一维numpy数组中提取一批索引的值。

    :param arr: 一维numpy数组。
    :param timesteps: 包含数组索引的张量。
    :param broadcast_shape: 更大的形状，具有K个维度，批次维度等于timesteps的长度。
    :return: 一个形状为[batch_size, 1, ...]的张量，其中形状具有K个维度。
    """
    # 将numpy数组转换成张量，并将其移到timesteps所在的设备上
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    
    # 调整res的维度以符合broadcast_shape的维度
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    
    # 将res扩展到broadcast_shape的形状
    return res.expand(broadcast_shape)
