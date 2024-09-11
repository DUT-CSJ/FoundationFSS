import torch.optim as optim


def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    pretrain_weight_list = []
    pretrain_bias_list =[]
    for name,value in model.named_parameters():
        if 'backbone' in name:
            if 'weight' in name:
                pretrain_weight_list.append(value)
            elif 'bias' in name:
                pretrain_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    # opt = optim.SGD([{'params': pretrain_weight_list, 'lr':lr},
    #                   {'params': pretrain_bias_list, 'lr':lr*2},
    #                   {'params': weight_list, 'lr':lr*10},
    #                   {'params': bias_list, 'lr':lr*20}], momentum=0.90, weight_decay=0.0005) # momentum = 0.99
    opt = optim.AdamW([{'params': pretrain_weight_list, 'lr':0.0005/10.},
                     {'params': pretrain_bias_list, 'lr':0.0005/5.},
                     {'params': weight_list, 'lr':0.0005*1},
                     {'params': bias_list, 'lr':0.0005*2}],  weight_decay=0.0)
    return opt


def adjust_learning_rate_poly(optimizer, iter, max_iter):
    base_lr = 0.0005
    reduce = ((1-float(iter)/max_iter)**(0.9))
    lr = base_lr * reduce
    optimizer.param_groups[0]['lr'] = lr