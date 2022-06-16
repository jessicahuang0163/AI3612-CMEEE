import re
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import MultipleLocator, tight_layout
from matplotlib import font_manager

log_file = '../logs/re-fgm-40.log'

loss, lr, epoch = [], [], []
eval_loss, eval_f1, eval_epoch = [], [], []

with open(log_file) as f:
    for line in f:
        if "{'loss':" in line:
            train_stat = re.search(r"{'loss': (.*), 'learning_rate': (.*), 'epoch': (.*)}", line, re.M | re.I)
            loss.append(float(train_stat.group(1)))
            lr.append(float(train_stat.group(2)))
            epoch.append(float(train_stat.group(3)))
        if "{'eval_loss'" in line:
            eval_stat = re.search(
                r"{'eval_loss': (.*), 'eval_f1': (.*), 'eval_runtime': (.*), 'eval_samples_per_second': (.*), 'eval_steps_per_second': (.*), 'epoch': (.*)}",
                line, re.M | re.I)
            eval_loss.append(float(eval_stat.group(1)))
            eval_f1.append(float(eval_stat.group(2)))
            eval_epoch.append(float(eval_stat.group(6)))

f.close()


def plot_line(x, y, ylabel):
    if not os.path.exists('../figs'):
        os.mkdir('../figs')
    colors = ['#0066FF', '#0099FF', '#00CCFF', '#00FFFF', '#00FFCC', '#00FFCC', '#00FFCC']
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    # axes.plot(epoch[25:], loss[25:], linestyle='-', color='#0066FF', linewidth=1.5)
    axes.plot(x, y, linestyle='-', color='#0066FF', linewidth=1.5)

    font = font_manager.FontProperties(family='serif', size=17)
    # 设置x、y轴标签
    axes.set_ylabel(ylabel, fontproperties=font, labelpad=12)
    axes.set_xlabel("Epoch", fontproperties=font, labelpad=12)

    # 设置y轴的刻度
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    # axes.set_xticks(np.arange(5, 45, 5))
    # axes.set_yticks(np.arange(0, 1.1, 0.1))

    # 画网格线
    axes.grid(which='major', c='lightgrey')
    # axes.legend(prop={'family':'serif', 'size':15})

    plt.savefig('../figs/{}.png'.format(ylabel.lower()), dpi=300)


plot_line(eval_epoch, eval_f1, 'Eval F1')
plot_line(eval_epoch, eval_loss, 'Eval loss')
plot_line(epoch[25:], loss[25:], 'Train Loss')
plot_line(epoch, lr, 'Learning rate (classifier)')
