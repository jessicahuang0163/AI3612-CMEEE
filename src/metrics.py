
from html import entities
import numpy as np
import torch
from collections import defaultdict
from typing import List, Union, NamedTuple, Tuple, Counter , Set
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK, W2_LABEL2ID, W2_ID2LABEL
class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        #'''NOTE: You need to finish the code of computing f1-score.
        pred_entities = extract_entities(predictions, for_nested_ner=False)
        label_entities = extract_entities(labels, for_nested_ner=False)

        equal_cnt, all_cnt = 0, 0

        for idx in range(len(pred_entities)):
            pred = pred_entities[idx]
            label = label_entities[idx]

            all_cnt = all_cnt + len(label) + len(pred)
            
            for entity in pred:
                if entity in label:
                    equal_cnt += 1
        #'''
        return { "f1": 2 * equal_cnt / all_cnt }


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.

        pred_entities1 = extract_entities(predictions[:, :, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, :, 1], for_nested_ner=True, first_labels=False)
        label_entities1 = extract_entities(labels1, for_nested_ner=True, first_labels=True)
        label_entities2 = extract_entities(labels2, for_nested_ner=True, first_labels=False)

        equal_cnt, all_cnt = 0, 0

        for idx in range(len(pred_entities1)):

            pred1, pred2 = pred_entities1[idx], pred_entities2[idx]
            label1, label2 = label_entities1[idx], label_entities2[idx]

            all_cnt = all_cnt + len(label1) + len(pred1) + len(label2) + len(pred2)
            
            for entity in pred1:
                if entity in label1:
                    equal_cnt += 1
            
            for entity in pred2:
                if entity in label2:
                    equal_cnt += 1                    
        # '''
        return { "f1": 2 * equal_cnt / all_cnt }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    for sent_id in range(batch_labels_or_preds.shape[0]):
        sent = batch_labels_or_preds[sent_id]
        entities = []
        begin = 0
        while id2label[sent[begin]] != NER_PAD:
            if id2label[sent[begin]][0] == 'B':
                end = begin
                while end < len(sent) - 1 and id2label[sent[end + 1]][0] == 'I':
                    end += 1

                id_list = sent[begin:end + 1]
                types = [id2label[id][2:] for id in id_list]
                types_cnt = Counter(types)
                max_freq = types_cnt.most_common(1)[0][1]
                max_freq_type = [(L, freq) for L, freq in types_cnt.items() if freq == max_freq]

                
                if len(max_freq_type) == 1:
                    entities.append((begin, end, max_freq_type[0][0]))
                else:
                    type_order = sorted([(L, _LABEL_RANK[L]) for (L, freq) in max_freq_type], \
                                key=lambda x:x[1], reverse=True)
                    entities.append((begin, end, type_order[0][0]))

            begin += 1
            if begin == len(sent): break

        batch_entities.append(entities)
    # '''
    return batch_entities



def decode_w2ner(w2ner_batch, batch_length):
    batch_entities = []
    w2ner_batch = w2ner_batch.detach().cpu().numpy()
    batch_length = batch_length.detach().cpu().numpy()
    for mat, len in zip(w2ner_batch, batch_length):
        cut = mat[:len, :len]
        NHW = defaultdict(list)  # Next-Neighboring-Word
        THW = defaultdict(set)   # Tail-Head-Word
        ent_type = defaultdict()

        for idx, idy in np.argwhere(cut == W2_LABEL2ID['suc']):
            idx, idy = idx.item(), idy.item()
            if idx < idy:
                NHW[idx].append(idy)

        for idx, idy in np.argwhere(cut != W2_LABEL2ID['suc']):
            idx, idy = idx.item(), idy.item()
            if idx >= idy and cut[idx, idy] != W2_LABEL2ID[NER_PAD]:
                THW[idy].add(idx)
                ent_type[(idy, idx)] = cut[idx, idy].item()
        pred = []

        def dfs(root, tails, entity= []):
            entity.append(root)
            if root in tails:
                pred.append(entity.copy())
            if root in NHW:
                for next_word in NHW[root]:
                    dfs(next_word, tails, entity)
            entity.pop()

        for h in THW:
            dfs(h, THW[h])

        entities = []
        for entity in pred:
            start, end = entity[0], entity[-1]
            type = W2_ID2LABEL[ent_type[(start, end)]]
            entities.append((start, end, type))

        batch_entities.append(entities)

    return batch_entities

class MetricsForW2NER:
    def __init__(self) -> None:
        self.true_pos = 0
        self.all_cnt = 0

    def accumulate(self, pred, labels, length):
        preds = decode_w2ner(pred, length)
        trues = decode_w2ner(labels, length)

        for pred, true in zip(preds, trues):
            pred = set(pred)
            true = set(true)
            for p in pred:
                if p in true:
                    self.true_pos += 1
            self.all_cnt = self.all_cnt + len(pred) + len(true)

    def _f1(self):
        return 2 * self.true_pos / self.all_cnt

    def summary(self):
        return {"f1": self._f1()}


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    