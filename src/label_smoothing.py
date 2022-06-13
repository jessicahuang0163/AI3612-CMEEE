import torch.nn as nn
import torch.nn.functional as F

from ee_data import (
    EE_label2id,
    EE_label2id1,
    EE_label2id2,
    EE_id2label1,
    EE_id2label2,
    EE_id2label,
    NER_PAD,
)

from transformers import Trainer


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.03, reduction="mean", ignore_index=EE_label2id[NER_PAD]):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )


class LabelSmoothingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss_fct = LabelSmoothingCrossEntropy()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
