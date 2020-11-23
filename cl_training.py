import os
from tensorboardX import SummaryWriter
import argparse
from argparse import RawTextHelpFormatter
import tqdm
import utils

import torch
from torch.autograd import Variable
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import torch.backends.cudnn as cudnn

from utils import load_weights_to_flatresnet

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='AutoPath Training', formatter_class=RawTextHelpFormatter)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', required=True, choices=['R18', 'R34', 'R50', 'R101'],
                    help="Select a model to test.\n"
                         "R18: ResNet18 based DeepLab V3;\n"
                         "R34: ResNet34 based DeepLab V3;\n"
                         "R50: ResNet18 based DeepLab V3;\n"
                         "R101: ResNet101 based DeepLab V3.")
parser.add_argument('--gpu', default='0', help='which GPU can be used')
parser.add_argument('--data_dir', required=True, help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--seg_model_checkpoint', required=True, default=None, help='checkpoint to load segmentation model')
parser.add_argument('--cv_dir', default='cv/task/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epoch_step', type=int, default=500, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--cl_step', type=int, default=1, help='steps for curriculum training')
parser.add_argument('--penalty', type=float, default=-50, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

writer = SummaryWriter(log_dir=os.path.join(args.cv_dir, 'logs'))

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)


def compute_VDC(input, target, smooth=1.):
    # assumes that input is a normalized probability
    if target.dim() == 4:
        target = target.unsqueeze(1)

    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = (input > 0.5).float()

    input = input.view(input.size(0), -1)
    target = target.view(input.size(0), -1)
    target = target.float()

    intersect = input * target
    denominator = input + target

    # preds|target:score  0|0:1   0|1:1/2   1|0:1/2   1|1:1
    # smooth for 0|0
    return (2. * intersect + smooth) / (denominator + smooth)


def compute_dice(input, target, smooth=1.):
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


def get_reward(preds, target, policy):
    block_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - block_use ** 2

    seg_score = compute_VDC(preds, target)

    dice = compute_dice(preds, target)

    right_set = (seg_score == 1).data

    sparse_tensor = sparse_reward.expand_as(right_set)

    penalty_tensor = args.penalty * torch.ones_like(sparse_tensor)

    reward = torch.where(right_set, sparse_tensor, penalty_tensor)
    reward = torch.mean(reward, -1)
    reward = reward.unsqueeze(1)

    return reward.float(), dice.float()


def train(epoch):
    dices, rewards, policies = [], [], []
    for batch_idx, (input, target) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        input, target = input, target.cuda()
        if not args.parallel:
            input = input.cuda()

        probs, value = agent(input)

        policy_map = probs.data.clone()

        policy_map[policy_map < 0.5] = 0.0
        policy_map[policy_map >= 0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs * args.alpha + (1 - probs) * (1 - args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        if args.cl_step < num_blocks:
            policy[:, :-args.cl_step] = 1
            policy_map[:, :-args.cl_step] = 1

            policy_mask = Variable(torch.ones(input.size(0), policy.size(1))).cuda()
            policy_mask[:, :-args.cl_step] = 0
        else:
            policy_mask = None

        seg_map = torch.sigmoid(seg_model.forward(input, policy_map))
        seg_sample = torch.sigmoid(seg_model.forward(input, policy))

        seg_map = F.interpolate(seg_map, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                                align_corners=True)
        seg_sample = F.interpolate(seg_sample, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                                   align_corners=True)

        reward_map, _ = get_reward(seg_map, target, policy_map.data)
        reward_sample, dice = get_reward(seg_sample, target, policy.data)

        advantage = reward_sample - reward_map

        loss = -distr.log_prob(policy)
        loss = loss * advantage.expand_as(policy)

        if policy_mask is not None:
            loss = policy_mask * loss  # mask for curriculum learning

        loss = loss.sum()

        probs = probs.clamp(1e-15, 1 - 1e-15)
        entropy_loss = -probs * torch.log(probs)
        entropy_loss = args.beta * entropy_loss.sum()

        loss = (loss - entropy_loss) / input.size(0)

        # ---------------------------------------------------------------------#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dices.append(dice.cpu())
        rewards.append(reward_sample.cpu())
        policies.append(policy.data.cpu())

    dice, reward, sparsity, variance, policy_set, policy_list = utils.performance_stats(policies, rewards, dices)

    log_str = 'TRAIN - E: %d | D: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d' % (
    epoch, dice, reward, sparsity, variance, len(policy_set))
    print(log_str)
    print("policy_list:", policy_list)

    writer.add_scalar('train_dice', dice, epoch)
    writer.add_scalar('train_reward', reward, epoch)
    writer.add_scalar('train_sparsity', sparsity, epoch)
    writer.add_scalar('train_variance', variance, epoch)
    writer.add_scalar('train_unique_policies', len(policy_set), epoch)


def test(epoch):
    dices, rewards, policies = [], [], []
    for batch_idx, (input, target) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        input, target = input, target.cuda()
        if not args.parallel:
            input = input.cuda()

        probs, _ = agent(input)

        policy = probs.data.clone()

        policy[policy < 0.5] = 0.0
        policy[policy >= 0.5] = 1.0
        policy = Variable(policy)

        if args.cl_step < num_blocks:
            policy[:, :-args.cl_step] = 1

        preds = torch.sigmoid(seg_model.forward(input, policy))
        preds = F.interpolate(preds, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                              align_corners=True)
        reward, dice = get_reward(preds, target, policy.data)

        dices.append(dice)
        rewards.append(reward)
        policies.append(policy.data)

    dice, reward, sparsity, variance, policy_set, policy_list = utils.performance_stats(policies, rewards, dices)

    log_str = 'TS - D: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d' % (dice, reward, sparsity, variance, len(policy_set))
    print(log_str)
    print("policy_list:", policy_list)

    writer.add_scalar('test_accuracy', dice, epoch)
    writer.add_scalar('test_reward', reward, epoch)
    writer.add_scalar('test_sparsity', sparsity, epoch)
    writer.add_scalar('test_variance', variance, epoch)
    writer.add_scalar('test_unique_policies', len(policy_set), epoch)

    # save the model
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()

    global best_dice
    if dice >= best_dice:
        state = {
            'agent': agent_state_dict,
            'epoch': epoch,
            'reward': reward,
            'dice': dice
        }
        torch.save(state, args.cv_dir + '/ckpt_E_%d_D_%.3f_R_%.2E_S_%.2f_#_%d.t7' % (
        epoch, dice, reward, sparsity, len(policy_set)))
        torch.save(state, args.cv_dir + '/best.t7')


best_dice = 0.0
if __name__ == '__main__':
    # define dateset
    train_ds = Dataset(os.path.join(args.data_dir, 'train', 'ct'), os.path.join(args.data_dir, 'train', 'seg'))
    test_ds = Dataset(os.path.join(args.data_dir, 'test', 'ct'), os.path.join(args.data_dir, 'test', 'seg'), test=True)

    # define data loader
    trainloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    seg_model, agent, num_blocks = utils.get_3dmodel(args.model)
    if args.seg_model_checkpoint != None:
        # load pretrained weights into flat ResNet
        seg_model_checkpoint = torch.load(args.seg_model_checkpoint)
        load_weights_to_flatresnet(seg_model_checkpoint, seg_model)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        agent.load_state_dict(checkpoint['agent'])
        start_epoch = checkpoint['epoch'] + 1
        print('loaded agent from', args.load)

    if args.parallel:
        agent = nn.DataParallel(agent)
        seg_model = nn.DataParallel(seg_model)

    seg_model.eval().cuda()
    agent.cuda()

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.wd)

    lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        lr_scheduler.adjust_learning_rate(epoch)

        if args.cl_step < num_blocks:
            args.cl_step = 1 + 1 * (epoch // 10)
        else:
            args.cl_step = num_blocks

        print('training the last %d blocks ...' % args.cl_step)
        train(epoch)

        with torch.no_grad():
            if epoch != 0 and epoch % 10 == 0:
                test(epoch)
