import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import font_manager

from collections import Counter


def set_axis_style(ax, labels):
    font = font_manager.FontProperties(family='serif', size=16)
    font_tick = font_manager.FontProperties(family='serif', size=12)

    ax.yaxis.set_tick_params(direction='out')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(1, len(labels) + 1), fontproperties=font_tick, labels=labels)
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.grid(which='major', c='lightgrey')
    ax.set_ylabel('Dataset', fontproperties=font, labelpad=12)
    ax.set_xlabel('Length', fontproperties=font, labelpad=12)


dataset = ['train', 'dev', 'test']
filename = ['../data/CBLUEDatasets/CMeEE/CMeEE_{}.json'.format(i) for i in dataset]

text_len_list = []
type_num = []
entity_len_list = []

for idx, file in enumerate(filename):

    with open(file, encoding="utf8") as f:
        data = json.load(f)

    text_len = []
    entity_types = []
    entity_len = []

    for i in range(len(data)):
        text_len.append(len(data[i]['text']))
        if idx != 2:
            for entity in data[i]['entities']:
                entity_types.append(entity['type'])
                entity_len.append(len(entity['entity']))

    text_len_list.append(text_len)
    type_num.append(entity_types)
    entity_len_list.append(entity_len)

##############      plot text length        ##############

if not os.path.exists('../figs'):
    os.mkdir('../figs')
fig, axes = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
parts = axes.violinplot(text_len_list, vert=False, positions=[3, 2, 1], showmedians=False)
set_axis_style(axes, ['test', 'dev', 'train'])
plt.savefig('../figs/text_len.png', dpi=300)

fig, axes = plt.subplots(1, 1, figsize=(8, 3), tight_layout=True)
parts = axes.violinplot(entity_len_list[:2], vert=False, positions=[2, 1], showmedians=False)
set_axis_style(axes, ['dev', 'train'])
plt.savefig('../figs/entity_len.png', dpi=300)

##############      plot type number        ##############

fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)

labels = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod']

train_type = Counter(type_num[0])
dev_type = Counter(type_num[1])

train_type_num = [train_type[i] for i in labels]
dev_type_num = [dev_type[i] for i in labels]

x = np.arange(1, 2 * len(labels) + 1, 2)  # the label locations
width = 0.5  # the width of the bars

rects1 = ax.bar(x - width / 2, train_type_num, width, label='Train', color='#11C2EE')
rects2 = ax.bar(x + width / 2, dev_type_num, width, label='Dev', color='#1AE694')

# Add some text for labels, title and custom x-axis tick labels, etc.
font = font_manager.FontProperties(family='serif', size=19)
font_tick = font_manager.FontProperties(family='serif', size=16)
ax.set_ylabel('Number', fontproperties=font, labelpad=15)
ax.set_xlabel('Labels', fontproperties=font, labelpad=15)
ax.set_xticks(x, labels, fontproperties=font_tick)
ax.tick_params(axis='y', labelsize=13)
ax.legend(prop={'family': 'serif', 'size': 17})
ax.grid(axis='y', which='major', c='lightgrey')

plt.savefig('../figs/type.png', dpi=400)
