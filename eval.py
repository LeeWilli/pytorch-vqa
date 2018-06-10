import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def run(net, v, q, q_len):
    """ Run an epoch over the given loader """
    net.eval()
    answ = []

    log_softmax = nn.LogSoftmax().cuda()

    var_params = {
        'volatile': True,
        'requires_grad': False,
    }
    v = Variable(v.cuda(async=True), **var_params)
    q = Variable(q.cuda(async=True), **var_params)
    q_len = Variable(q_len.cuda(async=True), **var_params)

    out = net(v, q, q_len)

    # store information about evaluation of this minibatch
    _, answer = out.data.cpu().max(dim=1)
    answ.append(answer.view(-1))

    if(len(answ)>0):
        answ = list(torch.cat(answ, dim=0))
    return answ


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
