import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import (
    EE_label2id,
    EE_label2id1,
    EE_label2id2,
    EE_id2label1,
    EE_id2label2,
    EE_id2label,
    NER_PAD,
    _LABEL_RANK,
)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER:  # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred

        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]

        #'''NOTE: You need to finish the code of computing f1-score.
        pred_true, pred_tot = 0, 0
        pred_entits = extract_entities(predictions, for_nested_ner=False)
        label_entits = extract_entities(labels, for_nested_ner=False)

        for idx in range(len(pred_entits)):
            pred = pred_entits[idx]
            label = label_entits[idx]

            pred_tot += len(label) + len(pred)

            for ent in pred:
                if ent in label:
                    pred_true += 1

        f1 = 2 * pred_true / pred_tot
        #'''

        return {"f1": f1}


class ComputeMetricsForNestedNER:  # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred

        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD]  # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]

        # '''NOTE: You need to finish the code of computing f1-score.
        pred_true, pred_tot = 0, 0
        pred_entit1 = extract_entities(
            predictions[:, :, 0], for_nested_ner=True, first_labels=True
        )
        pred_entit2 = extract_entities(
            predictions[:, :, 1], for_nested_ner=True, first_labels=False
        )
        label_entit1 = extract_entities(labels1, for_nested_ner=True, first_labels=True)
        label_entit2 = extract_entities(
            labels2, for_nested_ner=True, first_labels=False
        )

        for idx in range(len(pred_entit1)):
            pred1, pred2 = pred_entit1[idx], pred_entit2[idx]
            label1, label2 = label_entit1[idx], label_entit2[idx]

            pred_tot += len(label1) + len(pred1) + len(label2) + len(pred2)

            for ent in pred1:
                if ent in label1:
                    pred_true += 1

            for ent in pred2:
                if ent in label2:
                    pred_true += 1

            f1 = 2 * pred_true / pred_tot
        # '''

        return {"f1": f1}


def extract_entities(
    batch_labels_or_preds: np.ndarray,
    for_nested_ner: bool = False,
    first_labels: bool = True,
) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[
        NER_PAD
    ]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]

    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    for idx in range(batch_labels_or_preds.shape[0]):
        sentence = batch_labels_or_preds[idx]
        entities = []
        begin = 0
        for begin in range(len(sentence)):
            if (
                id2label[sentence[begin]][0] == "B"
                and id2label[sentence[begin]] != NER_PAD
            ):
                end = begin + 1
                while end < len(sentence) and id2label[sentence[end]][0] == "I":
                    end += 1
                # Now we've got the range of an entity.
                id_list = sentence[begin:end]
                entity_type = [id2label[id][2:] for id in id_list]
                max_freq = Counter(entity_type).most_common(1)[0][1]
                max_freq_entity_type = [
                    (L, freq)
                    for L, freq in Counter(entity_type).items()
                    if freq == max_freq
                ]

                if len(max_freq_entity_type) == 1:
                    entities.append((begin, end - 1, max_freq_entity_type[0][0]))
                else:
                    order = sorted(
                        [(L, _LABEL_RANK[L]) for (L, freq) in max_freq_entity_type],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    entities.append((begin, end - 1, order[0][0]))

        batch_entities.append(entities)
    # '''
    return batch_entities


if __name__ == "__main__":

    # Test for ComputeMetricsForNER
    predictions = np.load("../test_files/predictions.npy")
    labels = np.load("../test_files/labels.npy")

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics["f1"] - 0.606179116) < 1e-5:
        print("You passed the test for ComputeMetricsForNER.")
    else:
        print("The result of ComputeMetricsForNER is not right.")

    # Test for ComputeMetricsForNestedNER
    predictions = np.load("../test_files/predictions_nested.npy")
    labels1 = np.load("../test_files/labels1_nested.npy")
    labels2 = np.load("../test_files/labels2_nested.npy")

    metrics = ComputeMetricsForNestedNER()(
        EvalPrediction(predictions, (labels1, labels2))
    )

    if abs(metrics["f1"] - 0.60333644) < 1e-5:
        print("You passed the test for ComputeMetricsForNestedNER.")
    else:
        print("The result of ComputeMetricsForNestedNER is not right.")
