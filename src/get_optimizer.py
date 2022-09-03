from src.pachage_list import *

def get_optimizer_grouped_parameters(model, mode='all', learning_rate=2e-5, weight_decay=0.01, diff_rate=2.6):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = learning_rate
    no_decay = ['bias', 'gamma', 'beta']
    if mode == 'all':
        optimizer_parameters = filter(
            lambda x: x.requires_grad, model.parameters())
    elif mode == 'specific':
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': 1e-3,
             'weight_decay_rate':0.01}
        ]
    elif mode == 'layerwise':
        diff_rate = diff_rate
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group4 = ['layer.12.', 'layer.13.', 'layer.14.', 'layer.15.']
        group5 = ['layer.16.', 'layer.17.', 'layer.18.', 'layer.19.']
        group6 = ['layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']
        group_all = [
            'layer.0.', 'layer.1.', 'layer.2.', 'layer.3.',
            'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.', 'layer.8.', 'layer.9.',
            'layer.10.', 'layer.11.', 'layer.12.', 'layer.13.',
            'layer.14.', 'layer.15.', 'layer.16.', 'layer.17.',
            'layer.18.', 'layer.19.', 'layer.20.', 'layer.21.',
            'layer.22.', 'layer.23.'
        ]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(
                nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group1)], 'weight_decay': 0.01, 'lr': learning_rate/(diff_rate*4)},
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group2)], 'weight_decay': 0.01, 'lr': learning_rate/(diff_rate*3)},
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group3)], 'weight_decay': 0.01, 'lr': learning_rate/(diff_rate*2)},
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group4)], 'weight_decay': 0.01, 'lr': learning_rate/diff_rate},
            {'params': [p for n, p in model.model.named_parameters() if not any(
                nd in n for nd in no_decay) and any(nd in n for nd in group5)], 'weight_decay': 0.01, 'lr': learning_rate},
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group6)], 'weight_decay': 0.01, 'lr': learning_rate*diff_rate},
            {'params': [p for n, p in model.model.named_parameters() if any(
                nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group1)], 'weight_decay': 0.0, 'lr': learning_rate/(diff_rate*4)},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group2)], 'weight_decay': 0.0, 'lr': learning_rate/(diff_rate*3)},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group3)], 'weight_decay': 0.0, 'lr': learning_rate/(diff_rate*2)},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group4)], 'weight_decay': 0.0, 'lr': learning_rate/diff_rate},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group5)], 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group6)], 'weight_decay': 0.0, 'lr': learning_rate*diff_rate},
            {'params': [p for n, p in model.named_parameters() if "model" not in n], 'lr':1e-3, "weight_decay" : 0.09},
        ]
    return optimizer_parameters