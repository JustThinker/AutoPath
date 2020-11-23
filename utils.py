import math
import os
import shutil

import numpy as np
import torch


def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir + '/args.txt', 'w') as f:
        f.write(str(args))


def performance_stats(policies, rewards, dices):
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    dice = torch.cat(dices, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_list = [''.join(p) for p in policy_set]
    policy_set = set([''.join(p) for p in policy_set])

    return dice, reward, sparsity, variance, policy_set, policy_list


class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch % self.epoch_step == 0:
                print('# setting learning_rate to %.2E' % lr)


# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(source_model, target_model):
    # compatibility for nn.Modules + checkpoints
    if hasattr(source_model, 'seg_model'):
        source_model = {'seg_model': source_model.state_dict()}
    source_state = source_model['seg_model']
    # target_state = target_model.state_dict()

    # remove the module. prefix if it exists (thanks nn.DataParallel)
    if list(source_state.keys())[0].startswith('module.'):
        source_state = {k[7:]: v for k, v in source_state.items()}

    target_model.load_state_dict(source_state)
    return target_model


def load_checkpoint(net, load):
    checkpoint = torch.load(load)
    if 'seg_model' in checkpoint:
        net.load_state_dict(checkpoint['seg_model'])
        print('loaded resnet from', os.path.basename(load))
    if 'agent' in checkpoint:
        net.load_state_dict(checkpoint['agent'])
        print('loaded agent from', os.path.basename(load))


def get_3dmodel(model):
    from models import deeplab_3d, base_3d
    seg_model = None
    agent = None

    if model == 'R101':
        blocks = 33
        layer_config = [3, 4, 23, 3]
        seg_model = deeplab_3d.FlatDeeplab224(base_3d.Bottleneck, layer_config, num_classes=1)
        agent = deeplab_3d.Policy224([1, 1, 1, 1], num_blocks=blocks)
    elif model == 'R50':
        blocks = 16
        layer_config = [3, 4, 6, 3]
        seg_model = deeplab_3d.FlatDeeplab224(base_3d.Bottleneck, layer_config, num_classes=1)
        agent = deeplab_3d.Policy224([1, 1, 1, 1], num_blocks=blocks)
    elif model == 'R34':
        blocks = 16
        layer_config = [3, 4, 6, 3]
        seg_model = deeplab_3d.FlatDeeplab224(base_3d.BasicBlock, layer_config, num_classes=1)
        agent = deeplab_3d.Policy224([1, 1, 1, 1], num_blocks=blocks)
    elif model == 'R18':
        blocks = 8
        layer_config = [2, 2, 2, 2]
        seg_model = deeplab_3d.FlatDeeplab224(base_3d.BasicBlock, layer_config, num_classes=1)
        agent = deeplab_3d.Policy224([1, 1, 1, 1], num_blocks=blocks)

    return seg_model, agent, blocks


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        if not math.isnan(value):
            self.count += n
            self.sum += value * n
            self.avg = self.sum / self.count


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

        print(path + 'create succefully.')
        return True
    else:
        print(path + ' already existed.')
        return False