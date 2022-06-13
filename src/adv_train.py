import torch
import torch.nn as nn
from typing import Dict, Union, Any

from transformers import Trainer


# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=1.0):
        self.model = model.module if hasattr(model, "module") else model
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


class PGD:
    def __init__(self, model, eps=1.0, alpha=0.3):
        self.model = model.module if hasattr(model, "module") else model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name="word_embeddings", is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


# Overload the trainer with adversarial training
class Trainer_FGM(Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer_FGM, self).__init__(*args, **kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        fgm = FGM(model, eps=0.5)

        # with autocast(): # autocast_smart_context_manager
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = self.scaler.scale(loss)
        loss.backward()

        fgm.attack(emb_name="word_embeddings")
        loss_adv = self.compute_loss(model, inputs)
        loss_adv.backward()
        fgm.restore()

        return loss_adv.detach()


# Overload the trainer with adversarial training
class Trainer_PGD(Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer_PGD, self).__init__(*args, **kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # with autocast(): # autocast_smart_context_manager
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = self.scaler.scale(loss)
        loss.backward()

        pgd = PGD(model, eps=1.0, alpha=0.3)
        pgd_k = 3

        pgd.backup_grad()

        loss_adv = None
        for _t in range(pgd_k):
            pgd.attack(is_first_attack=(_t == 0))

            if _t != pgd_k - 1:
                model.zero_grad()
            else:
                pgd.restore_grad()

            loss_adv = self.compute_loss(model, inputs)
            loss_adv.backward()

        pgd.restore()

        return loss_adv.detach()
