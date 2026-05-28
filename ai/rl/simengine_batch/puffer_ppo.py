from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.distributions import Normal


@dataclass(frozen=True)
class PufferPPOConfig:
    total_timesteps: int
    horizon: int
    num_envs: int
    minibatch_size: int = 8192
    learning_rate: float = 0.015
    anneal_lr: bool = True
    min_lr_ratio: float = 0.0
    gamma: float = 0.995
    gae_lambda: float = 0.90
    replay_ratio: float = 1.0
    clip_coef: float = 0.2
    vf_coef: float = 2.0
    vf_clip_coef: float = 0.2
    max_grad_norm: float = 1.5
    ent_coef: float = 0.001
    beta1: float = 0.95
    beta2: float = 0.999
    eps: float = 1e-12
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    prio_alpha: float = 0.8
    prio_beta0: float = 0.2
    optimizer: str = "adam"

    @property
    def batch_size(self) -> int:
        return int(self.horizon) * int(self.num_envs)

    @property
    def total_epochs(self) -> int:
        return max(1, int(self.total_timesteps) // max(self.batch_size, 1))

    @property
    def minibatch_segments(self) -> int:
        if self.minibatch_size % self.horizon != 0:
            raise ValueError("Puffer PPO requires minibatch_size divisible by horizon")
        return int(self.minibatch_size) // int(self.horizon)


class PufferMLPPolicy(nn.Module):
    """Small continuous-action policy matching Puffer's Policy interface."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_size, hidden_size), nn.GELU()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.decoder_mean = nn.Linear(hidden_size, action_size)
        self.decoder_logstd = nn.Parameter(torch.zeros(1, action_size))
        self.value_function = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device | str):
        return ()

    def forward_eval(self, obs: torch.Tensor, state=()):
        hidden = self.encoder(obs.float())
        mean = self.decoder_mean(hidden)
        logstd = self.decoder_logstd.expand_as(mean)
        value = self.value_function(hidden).squeeze(-1)
        return Normal(mean, torch.exp(logstd)), value, state

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"expected [batch, time, obs], got {tuple(obs.shape)}")
        batch, time = obs.shape[:2]
        hidden = self.encoder(obs.reshape(batch * time, *obs.shape[2:]).float())
        mean = self.decoder_mean(hidden)
        logstd = self.decoder_logstd.expand_as(mean)
        value = self.value_function(hidden).reshape(batch, time)
        return Normal(mean, torch.exp(logstd)), value


class PufferPPO:
    """Adapted from puffer/pufferlib/torch_pufferl.py for external env runners."""

    def __init__(self, policy: nn.Module, config: PufferPPOConfig):
        self.policy = policy
        self.config = config
        self.epoch = 0
        self.optimizer = _make_optimizer(policy, config)
        self.ratio = None

    def update(self, rollout: dict[str, np.ndarray], device: torch.device) -> dict[str, float]:
        cfg = self.config
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=device).transpose(0, 1).contiguous()
        act = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=device).transpose(0, 1).contiguous()
        val = torch.as_tensor(rollout["values"], dtype=torch.float32, device=device).T.contiguous()
        lp = torch.as_tensor(rollout["logprobs"], dtype=torch.float32, device=device).T.contiguous()
        rew = torch.as_tensor(rollout["rewards"], dtype=torch.float32, device=device).T.contiguous().clamp(-1, 1)
        ter = torch.as_tensor(rollout["dones"], dtype=torch.float32, device=device).T.contiguous()

        if self.ratio is None or self.ratio.shape != val.shape or self.ratio.device != device:
            self.ratio = torch.ones_like(val, device=device)
        else:
            self.ratio.fill_(1.0)

        self._anneal_lr()
        anneal_beta = cfg.prio_beta0 + (1.0 - cfg.prio_beta0) * cfg.prio_alpha * self.epoch / cfg.total_epochs
        num_minibatches = max(1, int(cfg.replay_ratio * cfg.batch_size / cfg.minibatch_size))
        losses = defaultdict(float)
        advantages = torch.zeros_like(val)

        for _ in range(num_minibatches):
            advantages = puffer_advantage(
                val,
                rew,
                ter,
                self.ratio,
                cfg.gamma,
                cfg.gae_lambda,
                cfg.vtrace_rho_clip,
                cfg.vtrace_c_clip,
            )
            idx, mb_prio = _sample_priority_indices(
                advantages,
                cfg.prio_alpha,
                cfg.minibatch_segments,
                cfg.num_envs,
                anneal_beta,
            )

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            logits, newvalue = self.policy(mb_obs)
            _, newlogprob, entropy = sample_logits(logits, action=mb_actions)

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

            adv = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cfg.vf_clip_coef, cfg.vf_clip_coef)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss
            val[idx] = newvalue.detach().float()

            losses["policy_loss"] += pg_loss
            losses["value_loss"] += value_loss
            losses["entropy"] += entropy_loss
            losses["old_approx_kl"] += old_approx_kl
            losses["approx_kl"] += approx_kl
            losses["clipfrac"] += clipfrac
            losses["importance"] += ratio.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.epoch += 1
        out = {k: float(v.detach().item() / num_minibatches) for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        out["explained_variance"] = float("nan") if float(var_y) == 0.0 else float((1.0 - (y_true - y_pred).var() / var_y).item())
        return out

    def _anneal_lr(self) -> None:
        cfg = self.config
        if not cfg.anneal_lr or self.epoch <= 0:
            return
        lr_ratio = self.epoch / cfg.total_epochs
        lr_min = cfg.learning_rate * cfg.min_lr_ratio
        learning_rate = lr_min + 0.5 * (cfg.learning_rate - lr_min) * (1.0 + math.cos(math.pi * lr_ratio))
        self.optimizer.param_groups[0]["lr"] = learning_rate


def sample_logits(logits: Normal, action: torch.Tensor | None = None):
    batch = logits.loc.shape[0]
    if action is None:
        action = logits.sample().view(batch, -1)
    else:
        action = action.view(batch, -1)
    log_probs = logits.log_prob(action).sum(1)
    entropy = logits.entropy().view(batch, -1).sum(1)
    return action, log_probs, entropy


def puffer_advantage(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    importance: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
) -> torch.Tensor:
    advantages = torch.zeros_like(values)
    lastpufferlam = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    for t in range(values.shape[1] - 2, -1, -1):
        next_nonterminal = 1.0 - dones[:, t + 1]
        imp = importance[:, t]
        rho_t = torch.minimum(imp, torch.as_tensor(vtrace_rho_clip, dtype=values.dtype, device=values.device))
        c_t = torch.minimum(imp, torch.as_tensor(vtrace_c_clip, dtype=values.dtype, device=values.device))
        delta = rho_t * rewards[:, t + 1] + gamma * values[:, t + 1] * next_nonterminal - values[:, t]
        lastpufferlam = delta + gamma * gae_lambda * c_t * lastpufferlam * next_nonterminal
        advantages[:, t] = lastpufferlam
    return advantages


def _sample_priority_indices(
    advantages: torch.Tensor,
    prio_alpha: float,
    minibatch_segments: int,
    total_agents: int,
    anneal_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv = advantages.abs().sum(axis=1)
    prio_weights = torch.nan_to_num(adv ** prio_alpha, 0.0, 0.0, 0.0)
    prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
    idx = torch.multinomial(prio_probs, int(minibatch_segments), replacement=True)
    mb_prio = (total_agents * prio_probs[idx, None]) ** -anneal_beta
    return idx, mb_prio


def _make_optimizer(policy: nn.Module, config: PufferPPOConfig) -> torch.optim.Optimizer:
    if config.optimizer == "muon":
        return Muon(
            policy.parameters(),
            lr=config.learning_rate,
            momentum=config.beta1,
            eps=config.eps,
        )
    return torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


NS_COEFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def zeropower_via_newtonschulz5(grad: Tensor, eps: float = 1e-7) -> Tensor:
    x = grad.clone()
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    x = x / torch.clamp(grad.norm(dim=(-2, -1)), min=eps)
    for a, b, c in NS_COEFS:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    return x.to(grad.dtype)


class Muon(Optimizer):
    """Small local copy of puffer/pufferlib/muon.py."""

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"learning rate should be >= 0 but is {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is {weight_decay}")
        super().__init__(params, {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eps": eps,
        })

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.mul_(momentum)
                buf.add_(grad)
                grad = grad.add(buf * momentum)
                if grad.ndim >= 2:
                    original_shape = grad.shape
                    grad = grad.view(grad.shape[0], -1)
                    grad = zeropower_via_newtonschulz5(grad, eps)
                    grad *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
                    grad = grad.view(original_shape)
                param.mul_(1 - lr * weight_decay)
                param.sub_(lr * grad)
        return loss
