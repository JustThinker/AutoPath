import os
import argparse
from argparse import RawTextHelpFormatter
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
from losses import get_loss_criterion
from metrics import get_evaluation_metric

cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training Original Model for Segmentation', formatter_class=RawTextHelpFormatter)
parser.add_argument('--data_dir', required=True,
                    help="Path of train data directory. This directory must match the following:\n"
                         "[train data dir]\n"
                         "                -- [ct]\n"
                         "                -- [seg]\n")
parser.add_argument('--model', required=True, default='R50', choices=['R18', 'R34', 'R50', 'R101'],
                    help="Select a model to test.\n"
                         "R18: ResNet18 based DeepLab V3;\n"
                         "R34: ResNet34 based DeepLab V3;\n"
                         "R50: ResNet18 based DeepLab V3;\n"
                         "R101: ResNet101 based DeepLab V3.")
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--gpu', default='0', help='which GPU can be used')
parser.add_argument('--load', default=None, help='checkpoint to load model from')
parser.add_argument('--cv_dir', required=True, default='cv/task', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--epoch_step', type=int, default=100, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=400, help='total epochs to run')
parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

writer = SummaryWriter(log_dir=os.path.join(args.cv_dir, 'logs'))

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)


def train(epoch):
    loss_ave = utils.RunningAverage()
    dice_ave = utils.RunningAverage()
    for batch_idx, (input, target) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        input, target = input, target.cuda(async=True)
        if not args.parallel:
            input = input.cuda()

        seg = seg_model.forward(input, fully=True)

        seg = F.interpolate(seg, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                            align_corners=True)

        loss = loss_criterion(seg, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = evaluation_metric(seg, target)

        loss_ave.update(loss.item(), input.size(0))
        dice_ave.update(dice, input.size(0))

    log_str = 'TRAIN-------Epoch: %d | Loss: %.5f | Dice: %.5f' % (epoch, loss_ave.avg, dice_ave.avg)
    print(log_str)

    writer.add_scalar('train_loss', loss_ave.avg, epoch)
    writer.add_scalar('train_dice', dice_ave.avg, epoch)


def test(epoch):

    dice_ave = utils.RunningAverage()
    for batch_idx, (input, target) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        input, target = input.cuda(), target.cuda(async=True)
        if not args.parallel:
            input = input.cuda()

        seg = seg_model.forward_full(input)
        seg = F.interpolate(seg, size=(target.size(1), target.size(2), target.size(3)), mode="trilinear",
                            align_corners=True)

        dice = evaluation_metric(seg, target)

        dice_ave.update(dice, input.size(0))

    log_str = 'VAL-----Epoch: %d | Dice: %.5f' % (epoch, dice_ave.avg)
    print(log_str)

    writer.add_scalar('val_dice', dice_ave.avg, epoch)
    global best_dice
    if dice_ave.avg > best_dice:
        best_dice = dice_ave.avg
        # save the model
        state_dict = seg_model.module.state_dict() if args.parallel else seg_model.state_dict()

        state = {
            'seg_model': state_dict,
            'epoch': epoch,
        }
        torch.save(state, args.cv_dir + '/ckpt_E_%d_DICE_%.5f.t7' % (epoch, dice_ave.avg))


# --------------------------------------------------------------------------------------------------------#
best_dice = 0.0
if __name__ == '__main__':
    # define dateset
    train_ds = Dataset(os.path.join(args.data_dir, 'train', 'ct'), os.path.join(args.data_dir, 'train', 'seg'))
    test_ds = Dataset(os.path.join(args.data_dir, 'test', 'ct'), os.path.join(args.data_dir, 'test', 'seg'), test=True)

    # define data loader
    trainloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    testloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    seg_model, _, _ = utils.get_3dmodel(args.model)
    seg_model = seg_model.cuda()

    loss_criterion = get_loss_criterion('dice')
    evaluation_metric = get_evaluation_metric()

    start_epoch = 0
    if args.load is not None:
        utils.load_checkpoint(seg_model, args.load)
        start_epoch = 0
        print('loaded weight from', args.load)

    if args.parallel:
        seg_model = nn.DataParallel(seg_model)

    optimizer = optim.Adam(seg_model.parameters(), lr=args.lr, weight_decay=args.wd)

    lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        lr_scheduler.adjust_learning_rate(epoch)
        train(epoch)

        with torch.no_grad():
            if epoch != 0 and epoch % 10 == 0:
                print(args.cv_dir)
                test(epoch)
