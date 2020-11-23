import os
import numpy as np
import tqdm
import random
import argparse
from argparse import RawTextHelpFormatter
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from datasets.dataset import Dataset
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Testing', formatter_class=RawTextHelpFormatter)
parser.add_argument('--gpu', default='0', help='which GPU can be used')
parser.add_argument('--model', required=True, default='R50', choices=['R18', 'R34', 'R50', 'R101'],
                    help="Select a model to test.\n"
                         "R18: ResNet18 based DeepLab V3;\n"
                         "R34: ResNet34 based DeepLab V3;\n"
                         "R50: ResNet18 based DeepLab V3;\n"
                         "R101: ResNet101 based DeepLab V3.")
parser.add_argument('--mode', default='auto', choices=['auto', 'full', 'one', 'first', 'last', 'random'],
                    help="Select a test mode.\n"
                         "auto: test segmentation model with AutoPath;\n"
                         "full: test segmentation model fully;\n"
                         "one: drop the n-th block;\n"
                         "first: drop all blocks before the n-th;\n"
                         "last: drop all blocks after the n-th block;\n"
                         "random: drop n blocks randomly.")
parser.add_argument('--data_dir', required=True,
                    help="Path of test data directory. This directory must match the following:\n"
                         "[test data dir]\n"
                         "                -- [ct]\n"
                         "                -- [seg]\n")
parser.add_argument('--load_seg_model', default=None, required=True, help="checkpoint to load segmentation model from")
parser.add_argument('--load_agent', default=None, required=True, help="checkpoint to load agent from")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# P = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def compute_dice(input, target, smooth=1.):
    # input and target shapes must match
    if target.dim() == 4:
        target = target.unsqueeze(1)

    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = (input > 0.5).float()

    input = input.view(input.size(0), -1)
    target = target.view(input.size(0), -1)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    denominator = (input + target).sum(-1)

    return (2. * intersect + smooth) / (denominator + smooth)


def test(P=None, mode='auto'):
    dices, policies = [], []

    # make file
    path = 'nii_image'
    utils.mkdir(path)
    for batch_idx, (input, target) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        input, target = input.cuda(), target.cuda()

        if mode == 'auto':
            probs, _ = agent(input)

            policy = probs.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
        else:
            assert P is not None, f"P can not be None when mode is {mode}."
            policy = P

        seg_map = torch.sigmoid(seg_model.forward_single(input, policy.data.squeeze(0)))
        seg_map = F.interpolate(seg_map, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                                align_corners=True)

        dice = compute_dice(seg_map, target)

        # save image
        seg_map_numpy = seg_map.cpu().detach().numpy()

        seg_map_numpy_s = np.squeeze(seg_map_numpy)
        sitk_img = sitk.GetImageFromArray(seg_map_numpy_s)
        sitk.WriteImage(sitk_img, path + '/' + mode + str(batch_idx) + '.nii', True)

        dices.append(dice)
        policies.append(policy.data)

    dice, _, sparsity, variance, policy_set, policy_list = utils.performance_stats(policies, dices, dices)

    log_str = u'''
    Dice: %.6f
    Block Usage: %.3f \u00B1 %.3f
    Unique Policies: %d
    ''' % (dice, sparsity, variance, len(policy_set))

    print(log_str)
    print('policy_set', policy_set)
    print('policy_list', policy_list)
    print('dices', list(map(lambda x: x.item(), dices)))
    return dice  #


def run(mode='auto', block_total=16, iterations=500):
    '''
    debug and three manual dropping strategies
    :param mode: [‘auto', 'full', 'first', 'last', 'random']
                auto, test segmentation model with AutoPath;
                one, drop the n-th block;
                full, test full segmentation model;
                first, drop all blocks before the n-th;
                last, drop all blocks after the n-th block;
                random, drop n blocks randomly.
    :param block_total: the total of the blocks
    :param iteration: the number of repeated experiments when mode is 'random'
    :return:
    '''
    assert mode in ['auto', 'full', 'one', 'first', 'last', 'random'], "Not support this mode!"

    if mode == 'auto':
        _ = test(None, mode)
    elif mode == 'full':  # don't drop
        list_all = [1 for _ in range(block_total)]
        print(f"mode {mode}:", list_all)
        P = torch.Tensor([list_all])
        dice = test(P, mode)
        with open('dice_log_full.txt', mode='a') as file_txt:
            file_txt.write(f'The result of full -----> dice: {dice}' + '\n')
    elif mode == 'one':
        for N in range(block_total):
            list_all = [1 for _ in range(block_total)]
            list_all[N] = 0
            print(f"mode {mode}:", list_all)
            P = torch.Tensor([list_all])
            dice = test(P, mode)
            with open('dice_log_one.txt', mode='a') as file_txt:
                file_txt.write(f'The result of dropping the {N} block -----> dice: {dice}' + '\n')
    elif mode == 'first':  # drop first N blocks
        list_all = [1 for _ in range(block_total)]
        for N in range(block_total-1):
            list_all[N] = 0
            print(f"mode {mode}:", list_all)
            P = torch.Tensor([list_all])
            dice = test(P, mode)
            with open('dice_log_first.txt', mode='a', encoding='utf-8') as file_txt:
                log_line = 'The result of dropping first {:2d} blocks -----> dice: {:.4f}'.format(N+1, dice)
                file_txt.write(log_line + '\n')
    elif mode == 'last':  # drop lst N blocks
        list_all = [1 for _ in range(block_total)]
        for N in range(block_total-1):
            list_all[block_total - 1 - N] = 0
            print(f"mode {mode}:", list_all)
            P = torch.Tensor([list_all])
            dice = test(P, mode)
            with open('dice_log_last.txt', mode='a') as file_txt:
                log_line = 'The result of dropping last {:2d} blocks -----> dice: {:.4f}'.format(N+1, dice)
                file_txt.write(log_line + '\n')
    elif mode == 'random':  # random drop N blocks
        for N in range(1, block_total):
            dice_list = []
            for i in range(iterations):
                list0 = [0 for _ in range(N)]
                list1 = [1 for _ in range(block_total - N)]
                list_all = list0 + list1
                random.shuffle(list_all)
                print(f"mode {mode} {i}/{iterations}:", list_all)
                P = torch.Tensor([list_all])
                dice = test(P, mode)
                dice_list.append(dice)
                with open('dice_log_random.txt', mode='a', encoding='utf-8') as file_txt:
                    file_txt.write(f'The result of iter{i} when random drop {N} blocks ---> dice: {dice}' + '\n')
            with open('dice_log_random.txt', mode='a', encoding='utf-8') as file_txt:
                file_txt.write(f'The max dice when drop {N} blocks randomly -----> dice: {max(dice_list)}' + '\n')


if __name__ == '__main__':
    # define dateset
    test_ds = Dataset(os.path.join(args.data_dir, 'ct'), os.path.join(args.data_dir, 'seg'), test=True)
    testloader = DataLoader(test_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
    seg_model, agent, blocks = utils.get_3dmodel(args.model)

    # if no model is loaded, use all blocks
    agent.logit.weight.data.fill_(0)
    agent.logit.bias.data.fill_(10)

    print("loading checkpoints")

    if args.load_seg_model is not None:
        utils.load_checkpoint(seg_model, args.load_seg_model)
    if args.load_agent is not None and args.mode == 'auto':
        utils.load_checkpoint(agent, args.load_agent)

    seg_model.eval().cuda()
    agent.eval().cuda()

    run(mode=args.mode, block_total=blocks, iterations=1)
