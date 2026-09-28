"""D3PM diffusion model as a PyTorch Lightning module."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler
import pytorch_lightning as pl

from dit import DDiT_Llama, DiTConfig


@dataclass
class D3PMConfig:
    n_T: int = 1000
    hybrid_loss_coeff: float = 0.001
    dit: DiTConfig = field(default_factory=DiTConfig)


class D3PM(nn.Module):
    def __init__(self, x0_model: nn.Module, n_T: int, num_classes: int, hybrid_loss_coeff: float) -> None:
        super().__init__()
        self.x0_model = x0_model
        self.n_T = n_T
        self.num_classes = num_classes
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999)

        q_onestep_mats = []
        for beta in self.beta_t:
            mat = torch.ones(num_classes, num_classes) * beta / num_classes
            mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            q_onestep_mats.append(mat)
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(1, 2)

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)

        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (self.n_T, num_classes, num_classes), self.q_mats.shape

    def _at(self, a, t, x):
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps)
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), (
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 1, x_0_logits, out)
        return bc

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1) - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond):
        predicted_x0_logits = self.x0_model(x_0, t, cond)
        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        x_t = self.q_sample(x, t, torch.rand((*x.shape, self.num_classes), device=x.device))
        assert x_t.shape == x.shape, f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"

        predicted_x0_logits = self.model_predict(x_t, t, cond)

        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)
        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * vb_loss + ce_loss

    def p_sample(self, x, t, cond, noise):
        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)
        return sample

    def sample(self, x, cond=None, return_all_samples=False):
        all_samples = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(x, t, cond, torch.rand((*x.shape, self.num_classes), device=x.device))
            all_samples.append(x)
        if return_all_samples:
            return torch.stack(all_samples)
        return x


class D3PMLitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        x0_model = DDiT_Llama.from_config(cfg.model.dit)
        self.d3pm = D3PM(
            x0_model,
            n_T=cfg.model.n_T,
            num_classes=cfg.model.dit.N,
            hybrid_loss_coeff=cfg.model.hybrid_loss_coeff,
        )
        self.loss_ema = None

    def forward(self, x, cond=None):
        return self.d3pm(x, cond=cond)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        cond = torch.stack([batch["cond_amp"], batch["cond_hemo"]], dim=1)
        loss = self.d3pm(input_ids, cond=cond)

        loss_value = loss.detach()
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss_value

        self.log("train_loss", self.loss_ema, on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.d3pm.x0_model.parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(self.cfg.train.warmup_steps),
            num_training_steps=max(num_training_steps, int(self.cfg.train.warmup_steps) + 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def sample(self, x, cond=None, return_all_samples=False):
        was_training = self.training
        self.eval()
        outputs = self.d3pm.sample(x, cond=cond, return_all_samples=return_all_samples)
        if was_training:
            self.train()
        return outputs

    def load_d3pm_state_dict(self, state_dict):
        """Load a raw D3PM state_dict (as saved by amp.py) or a Lightning checkpoint."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(key.startswith("d3pm.") for key in state_dict):
            self.load_state_dict(state_dict)
        else:
            self.d3pm.load_state_dict(state_dict)
