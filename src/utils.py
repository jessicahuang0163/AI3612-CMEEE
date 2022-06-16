import transformers


def bert_base_AdamW_LLRD(model):
    '''https://zhuanlan.zhihu.com/p/412889866'''

    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.weight"]
    init_lr = 9e-6  # 2e-5
    head_lr = 1e-3
    lr = init_lr

    # === Pooler and classifier ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "classifier" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 9e-6}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================

    for layer in range(11, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 9e-6}
        opt_parameters.append(layer_params)

        # lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 9e-6}
    opt_parameters.append(embed_params)

    return transformers.AdamW(opt_parameters, lr=init_lr, eps=1e-8)


def get_optimizer_and_scheduler_for_w2ner(updates_total, model):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.weight"]
    init_lr = 5e-6
    head_lr = 1e-3
    lr = init_lr

    # === Pooler and classifier ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "classifier" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 3e-6}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================

    for layer in range(11, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 3e-6}
        opt_parameters.append(layer_params)

        # lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 3e-6}
    opt_parameters.append(embed_params)

    optimizer = transformers.AdamW(opt_parameters, lr=1e-3)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * updates_total,
                                                             num_training_steps=updates_total)

    return (optimizer, scheduler)
