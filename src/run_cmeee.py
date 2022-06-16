import json
import os
import copy
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List, Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import LSTM
from transformers import (AdamW, BertLayer, BertTokenizer, HfArgumentParser,
                          Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)
from transformers.models.bert.modeling_bert import (BertAttention,
                                                    BertEmbeddings)
from w2ner import BertForW2NER
from w2ner_dataloader import W2_NUM_LABELS, W2NERDataset, fn_N2WER
from w2ner_trainer import W2NERTrainer

from adv_train import Trainer_FGM, Trainer_PGD
from args import CBLUEDataArgs, ModelConstructArgs
from ee_data import (EE_NUM_LABELS, EE_NUM_LABELS1, EE_NUM_LABELS2, NER_PAD,
                     CollateFnForEE, EE_label2id, EE_label2id1, EE_label2id2,
                     EEDataset)
from ensemble import Trainer_FGM_ensemble
from logger import get_logger
from metrics import (ComputeMetricsForNER, ComputeMetricsForNestedNER,
                     MetricsForW2NER, extract_entities)
from model import (BertForBiLSTMHeadNestedNER, BertForCRFHeadNER,
                   BertForCRFHeadNestedNER, BertForLinearHeadNER,
                   BertForLinearHeadNestedNER, BiLSTMClassifier, CRFClassifier,
                   LinearClassifier)
from utils import bert_base_AdamW_LLRD, get_optimizer_and_scheduler_for_w2ner

MODEL_CLASS = {
    'linear': BertForLinearHeadNER,
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested': BertForCRFHeadNestedNER,
    'bilstm_nested': BertForBiLSTMHeadNestedNER,
    'w2ner': BertForW2NER
}


def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_class = MODEL_CLASS[model_args.head_type]

    if 'nested' not in model_args.head_type:
        if model_args.head_type == 'w2ner':
            model = model_class.from_pretrained(model_args.model_path, num_labels1=W2_NUM_LABELS)
        else:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1,
                                            num_labels2=EE_NUM_LABELS2)

    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, title, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, title), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(title + " saved")


def generate_testing_results_w2ner(train_args, logger, predictions, test_dataset, title):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    final_answer = []
    for pred, example in zip(predictions, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in pred:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(os.path.join(train_args.output_dir, title), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(title + " saved")


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    swa_model = copy.deepcopy(model)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        if model_args.head_type == 'w2ner':
            train_dataset = W2NERDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer,
                                         for_nested_ner=for_nested_ner)
            dev_dataset = W2NERDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer,
                                       for_nested_ner=for_nested_ner)
        else:
            train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer,
                                      for_nested_ner=for_nested_ner)
            dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer,
                                    for_nested_ner=for_nested_ner)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    if model_args.head_type == 'w2ner':
        compute_metrics = MetricsForW2NER()
        collate_fn = fn_N2WER(tokenizer.pad_token_id)
    else:
        compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    if model_args.head_type == 'w2ner':

        batch_size = train_args.gradient_accumulation_steps * train_args.per_device_train_batch_size
        updates_total = len(train_dataset) * train_args.num_train_epochs // batch_size

        trainer = W2NERTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            optimizers=get_optimizer_and_scheduler_for_w2ner(updates_total, model),
        )
    else:
        trainer = Trainer_FGM_ensemble(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            optimizers=(bert_base_AdamW_LLRD(model), None)
        )

    # for n, p in list(model.named_parameters()):
    #     print(n)

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        if model_args.head_type == 'w2ner':
            test_dataset = W2NERDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer,
                                        for_nested_ner=for_nested_ner)
        else:
            trainer.swa(swa_model, train_args)
            test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer,
                                     for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        if model_args.head_type == 'w2ner':
            generate_testing_results_w2ner(train_args, logger, predictions, test_dataset, title='CMeEE_test.json')
        else:
            generate_testing_results(train_args, logger, predictions, test_dataset, title='CMeEE_test.json',
                                     for_nested_ner=for_nested_ner)

        # predictions, _labels, _metrics = trainer.predict(dev_dataset, metric_key_prefix="val")
        # generate_testing_results(train_args, logger, predictions, dev_dataset, title='CMeEE_dev.json', for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
