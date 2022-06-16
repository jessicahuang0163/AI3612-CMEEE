import collections
import math
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.utils import logging

from metrics import MetricsForW2NER, decode_w2ner

logger = logging.get_logger(__name__)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class W2NERTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "test") -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        prediction_loss_only = self.args.prediction_loss_only
        # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False)
        preds = []
        model.eval()
        num_samples = 0
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)
                num_samples += len(inputs['input_ids'])
                preds.extend(decode_w2ner(logits, inputs['length']))
        metrics = {}
        # metrics = denumpify_detensorize(metrics)
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size)))

        self._memory_tracker.stop_and_update_metrics(metrics)

        return PredictionOutput(predictions=preds, metrics=metrics, label_ids=None)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        model = self._wrap_model(self.model, training=False)

        batch_size = self.args.eval_batch_size if self.args.eval_batch_size is not None else dataloader.batch_size
        prediction_loss_only = self.args.prediction_loss_only

        logger.info("***** Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        observed_num_examples = 0
        eva_loss = 0.
        f1_metric = MetricsForW2NER()
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                observed_batch_size = len(inputs['input_ids'])
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)
                eva_loss += loss.item() * batch_size
                f1_metric.accumulate(logits, labels, inputs['length'])

        metrics = f1_metric.summary()

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=observed_num_examples,
                num_steps=math.ceil(observed_num_examples / total_batch_size),
            )
        )
        metrics['eval_loss'] = eva_loss / observed_num_examples

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # self._memory_tracker.stop_and_update_metrics(metrics)
        # self.log(metrics)

        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics
