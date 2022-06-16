import torch
import torch.nn as nn
import os
import copy

from transformers import Trainer
from adv_train import Trainer_FGM, Trainer_PGD


def get_model_path_list(base_dir):
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'pytorch_model.bin' == _file:
                model_lists.append(os.path.join(root, _file))
    # model_lists = sorted(model_lists,
    #                      key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists


# Overload the trainer with model ensemble
class Trainer_FGM_ensemble(Trainer_FGM):

    def __init__(self, *args, **kwargs):
        super(Trainer_FGM_ensemble, self).__init__(*args, **kwargs)

    def swa(self, model, train_args):

        model_path_list = get_model_path_list(train_args.output_dir)

        swa_model = copy.deepcopy(self.model)
        model.to(self.model.device)
        swa_n = 0.

        with torch.no_grad():
            for _ckpt in model_path_list:
                model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cuda')))
                tmp_para_dict = dict(model.named_parameters())

                alpha = 1. / (swa_n + 1.)

                for name, para in swa_model.named_parameters():
                    para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

                swa_n += 1

        # use 100000 to represent swa to avoid clash
        swa_model_dir = os.path.join(train_args.output_dir, f'checkpoint-100000')
        if not os.path.exists(swa_model_dir):
            os.mkdir(swa_model_dir)

        swa_model_path = os.path.join(swa_model_dir, 'pytorch_model.bin')
        torch.save(swa_model.state_dict(), swa_model_path)

        self.model = swa_model


if __name__ == '__main__':

    model_lists = get_model_path_list('../ckpts/debug')
    print(model_lists)
