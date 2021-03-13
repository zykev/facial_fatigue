from __future__ import print_function
seed = 4603
print('random seed: {}'.format(seed))
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from Code_tomse2 import load_materials, util, Model_Parts, pytorchtools
import time
from datetime import datetime
import pdb

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--pf', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--is_test', default=True, dest='is_test',
                    help='testing when is traing (default: True)')
parser.add_argument('--is_pretreat', default=False, dest='is_pretreat',
                    help='pretreating when is traing (default: False)')
parser.add_argument('--accumulation_step', default=1, type=int, metavar='M',
                    help='accumulation_step')
parser.add_argument('--loss_alpha', default=0.1, type=float,
                    help='adjust loss for crossentrophy')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of categorical classes')
parser.add_argument('--first_channel', default=64, type=int,
                    help='number of channel in first convolution layer in resnet')
parser.add_argument('--non_local_pos', default=3, type=int,
                    help='the position to add non_local block')
parser.add_argument('--batch_size', default=8, type=int,
                    help='batch size')


best_prec_total1 = 10
best_prec_mse1 = 10
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''MyNote '''


def abs_double(input, target):
    return abs(input - target)

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self,input,target):
        return abs_double(input, target)

#Label smoothing

class CrossEntropyLoss_label_smooth(nn.Module):
    def __init__(self, num_classes=10, smoothing=0.1):
        super(CrossEntropyLoss_label_smooth, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, outputs, targets):

        N = targets.size(0)
        # torch.Size([8, 10])
        # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
        smoothed_labels = torch.full(size=(N, self.num_classes), fill_value=self.smoothing / (self.num_classes - 1)).to(device)

        targets = targets.data
        # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
        smoothed_labels.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1 - self.smoothing)
        # 调用torch的log_softmax
        log_prob = nn.functional.log_softmax(outputs, dim=1)
        # 用之前得到的smoothed_labels来调整log_prob中每个值
        loss = - torch.sum(log_prob * smoothed_labels) / N

        return loss


class CrossEntropyLoss_bootstraps(nn.Module):
    def __init__(self, num_classes=10, weight_lmda=0.5):
        super(CrossEntropyLoss_bootstraps, self).__init__()
        self.num_classes = num_classes
        self.weight_lmda = weight_lmda

    def forward(self, outputs, targets, epoch):
        N = targets.size(0)
        targets_weight = (1 - epoch / args.epochs) ** self.weight_lmda
        # get predicted labels
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)

        log_prob = nn.functional.log_softmax(outputs, dim=1)

        # onehot index matrix of targets and pred
        onehot_targets = torch.zeros(size=(N, self.num_classes)).to(device)
        targets = targets.data
        onehot_targets.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1)

        onehot_pred = torch.zeros(size=(N, self.num_classes)).to(device)
        onehot_pred.scatter_(dim=1, index=pred, value=1)

        weight_labels = targets_weight * onehot_targets + (1 - targets_weight) * onehot_pred

        loss = - torch.sum(log_prob * weight_labels) / N

        return loss


def load_model(dir_model):
    MSEcriterion = nn.MSELoss()


    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=1, feature_dim=512, at_type='nonLocal')
    model = Model_Parts.LoadParameter(model, dir_model)

    model1 = torch.nn.DataParallel(Model_Parts.FullModel_Loss(model, MSEcriterion))
    return model1

def get_path(lr, wd):
    save_name = datetime.now().strftime('%m-%d_%H-%M')
    folder = 'lr'+str(lr)+'wd'+str(wd) + '_' + save_name
    if os.path.exists(folder):
        print("There is the folder")
        folder = folder + '_c'
        os.mkdir(folder)
    else:
        os.mkdir(folder)

    return folder


new_folder = get_path(args.lr, args.weight_decay)
early_stopping = pytorchtools.EarlyStopping(patience=20, path=new_folder+'/checkpoint.pt')

def main():
    global args, best_prec_total1, best_prec_mse1, device
    accumulation_step = args.accumulation_step
    print('epochs', args.epochs)
    print('learning rate:', args.lr)
    print('weight decay:', args.weight_decay)
    print('accumulation_step:', accumulation_step)
    print('loss alpha:', args.loss_alpha)
    print('num classes:', args.num_classes)
    print('first_channel', args.first_channel)
    print('non_local_pos', args.non_local_pos)
    print('batch_size:', args.batch_size)
    print('is_pretreat:', args.is_pretreat)


    # save model superparameter
    with open(os.path.join(new_folder, "hyperparam.txt"), "a+") as f:
        f.write('epochs' + ' ' + str(args.epochs) + '\n')
        f.write('learning rate' + ' ' + str(args.lr) + '\n')
        f.write('weight decay' + ' ' + str(args.weight_decay) + '\n')
        f.write('accumulation step' + ' ' + str(accumulation_step) + '\n')
        f.write('loss alpha' + ' ' + str(args.loss_alpha) + '\n')
        f.write('num classes' + ' ' + str(args.num_classes) + '\n')
        f.write('first channel' + ' ' + str(args.first_channel) + '\n')
        f.write('non local pos' + ' ' + str(args.non_local_pos) + '\n')
        f.write('batch size' + ' ' + str(args.batch_size) + '\n')
        f.write('is pretreat' + ' ' + str(args.is_pretreat) + '\n')

    dir_model = r"./model/epoch51_69.0"

    ''' Load data '''

    arg_rootTrain = r'/home/xiaotao/Desktop/Data-S375-cut224'
    arg_listTrain = r'./Data/376Data-Train.txt'
    arg_rooteval = r'/home/xiaotao/Desktop/Data-S375-cut224'
    arg_listeval = r'./Data/376Data-Eval.txt'

    # arg_rootTrain = r'/home/biai/BIAI/mood/Data-S375-cut224/'
    # arg_listTrain = r'./Data/376Data-Train.txt'
    # arg_rooteval = r'/home/biai/BIAI/mood/Data-S375-cut224/'
    # arg_listeval = r'./Data/376Data-Eval.txt'

    train_loader, val_loader = load_materials.LoadVideoAttention(arg_rootTrain, arg_listTrain, arg_rooteval,
                                                                      arg_listeval, batch_size=args.batch_size)
    ''' Eval '''
    print('args.evaluate', args.evaluate)
    loopTest = False
    if args.evaluate:
        if loopTest == True:
            dir_path = r"F:\Documents\biai-release_test\model"
            for mod in os.listdir(dir_path):
                print('-'*10, mod)
                dir_model = dir_path + "/" + mod
                model = load_model(dir_model)
                validate(val_loader, model)
        else:

            model = load_model(dir_model)
            validate(val_loader, model)

        return

    first_channel = args.first_channel
    feature_dim = first_channel * 4

    ''' Load model '''
    criterion1 = nn.CrossEntropyLoss()
    #criterion1 = CrossEntropyLoss_label_smooth(num_classes=10, smoothing=0.1)
    # criterion1 = CrossEntropyLoss_bootstraps(num_classes=args.num_classes, weight_lmda=0.5)
    criterion2 = nn.MSELoss()
    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=args.num_classes, feature_dim=feature_dim, non_local_pos=args.non_local_pos,
                                                         first_channel=first_channel)
    if args.is_pretreat:
        print("pretreat.............")
        model = Model_Parts.LoadParameter(model, dir_model)
       
        
    model = torch.nn.DataParallel(model)

    ''' Loss & Optimizer '''
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                weight_decay=args.weight_decay)
    
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    cudnn.benchmark = True



    ''' Train & Eval '''
    # print('args.evaluate', args.evaluate)
    # if args.evaluate:
    #     validate(val_loader, model)
    #     return

    for epoch in range(args.epochs):
        util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)

        print('...... Beginning Train epoch: {} ......'.format(epoch))
        totalloss, mseloss, ceacc = train(train_loader, model, criterion1, criterion2, args.loss_alpha, optimizer, epoch, accumulation_step)
        print('...... Beginning Test epoch: {} ......'.format(epoch))
        if args.is_test :
            totalloss1, mseloss1, ceacc1 = validate(val_loader, model, criterion1, criterion2, args.loss_alpha)
        else:
            totalloss1, mseloss1, ceacc1 = totalloss, mseloss, ceacc
        is_better = totalloss1 < best_prec_total1 and mseloss1 < best_prec_mse1
        
        if epoch % 100 == 99:
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'totalloss': totalloss1,
                'mseloss': mseloss1,
                'acc': ceacc1,
            }, path=new_folder)
        
        if is_better:
            print('better model!')
            best_prec_total1 = min(totalloss1, best_prec_total1)
            best_prec_mse1 = min(mseloss1, best_prec_mse1)
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'totalloss': totalloss1,
                'mseloss': mseloss1,
                'acc': ceacc1,
            }, path=new_folder)
        else:
            print('Model too bad & not save')
            
        with open(os.path.join(new_folder, "result.txt"), "a+") as f:
            f.write(str(round(totalloss,3))+" "+str(round(totalloss1,3))+" "+str(round(mseloss,3))+" "+str(round(mseloss1,3))+" "+str(round(ceacc,3))+" "+str(round(ceacc1,3)))
            f.write("\n")
            f.flush()
        if early_stopping.early_stop:
            print("Early stopping")
            break

    util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'totalloss': totalloss1,
                'mseloss': mseloss1,
                'acc': ceacc1,
            }, path=new_folder)
            

def train(train_loader, model, criterion1, criterion2, loss_alpha, optimizer, epoch, accumulation_step):
    global record_

    losses = util.AverageMeter()
    data_time = util.AverageMeter()
    accuracies = util.AverageMeter()
    # switch to train mode
    model.train()
    num_of_data = 0
    end = time.time()
    score_list = []
    for batch_idx, (input_image, sample) in enumerate(train_loader):

        data_time.update(time.time() - end)

        sample_catego = util.label_to_categorical(sample, args.num_classes)
        sample = sample.to(device)
        sample_catego = sample_catego.to(device)

        input_var = torch.autograd.Variable(input_image).permute((0, 2, 1, 3, 4))
        input_var = input_var.to(device)

        outputs = model(input_var)

        fatigue_loss_ce = criterion1(outputs, sample_catego)
        outputs_cont = util.output_tomse(outputs, args.num_classes)
        fatigue_loss_mse = criterion2(outputs_cont, sample)

        acc = util.calculate_accuracy(outputs, sample_catego)
        
        loss = loss_alpha * fatigue_loss_ce + fatigue_loss_mse
        compact = torch.tensor([fatigue_loss_mse])
        # compute output
        # loss, compact, frame_outputs, groud_truth = model(input_var, fatigue)
        score_list.append(compact)
        losses.update(loss.item(), input_var.size(0))
        ''' multi-task: log_vars weight '''
        # loss = loss.sum()
        loss = loss / accumulation_step
        loss.backward();

        accuracies.update(acc, input_var.size(0))

        num_of_data += len(input_var[0])
        # print(num_of_data)
        if 0 == batch_idx % accumulation_step:
            num_of_data = 0
            ''' model & full_model'''
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            
        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'
                .format(
                epoch, batch_idx, len(train_loader),
                data_time=data_time, loss=losses, acc=accuracies))

    ''' Compute Loss '''
    sum_loss = torch.stack(score_list, dim=0)
    max_loss, _ = sum_loss.max(0)
    min_loss, _ = sum_loss.min(0)
    mean_loss = sum_loss.mean(0)
    print('    Emo, Ene, Fat, Att, Mot, Glo')
    print(' Max Loss:  {}'.format(max_loss.cpu().detach().numpy()))
    print(' Min Loss:  {}'.format(min_loss.cpu().detach().numpy()))
    print(' Mean Loss: {}'.format(mean_loss.cpu().detach().numpy()))
    print(' Average Loss1(total loss): {}'.format(round(float(losses.avg), 4)))
    print(' Average Loss2(mse loss): {}'.format(round(float(sum_loss.mean()), 4)))
    print(' Average accuracy: {}'.format(round(float(accuracies.avg), 3)))

    return losses.avg, float(sum_loss.mean()), accuracies.avg


def validate(val_loader, model, criterion1, criterion2, loss_alpha):
    global record_
    losses = util.AverageMeter()
    data_time = util.AverageMeter()
    accuracies = util.AverageMeter()
    
    # switch to train mode
    end = time.time()
    score_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_image, sample) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            sample_catego = util.label_to_categorical(sample, args.num_classes)
            sample = sample.to(device)
            sample_catego = sample_catego.to(device)

            input_var = torch.autograd.Variable(input_image).permute((0, 2, 1, 3, 4))
            input_var = input_var.to(device)

            outputs = model(input_var)

            # compute output
            # loss, compact, frame_outputs, groud_truth = model(input_var, emotion, energy, fatigue, attention, motivate,
            #                                                   Global_Status)

            fatigue_loss_ce = criterion1(outputs, sample_catego)
            outputs_cont = util.output_tomse(outputs, args.num_classes)
            fatigue_loss_mse = criterion2(outputs_cont, sample)


            acc = util.calculate_accuracy(outputs, sample_catego)

            loss = loss_alpha * fatigue_loss_ce + fatigue_loss_mse

            compact = torch.tensor([fatigue_loss_mse])

            score_list.append(compact)
            ''' multi-task: log_vars weight '''
            #loss = loss.sum()
            accuracies.update(acc, input_var.size(0))
            losses.update(loss.item(), input_var.size(0))
            

            if batch_idx % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'
                    .format(
                    batch_idx, len(val_loader),
                    data_time=data_time, loss=losses, acc=accuracies))

    ''' Compute Loss '''
    sum_loss = torch.stack(score_list, dim=0)
    max_loss, _ = sum_loss.max(0)
    min_loss, _ = sum_loss.min(0)
    mean_loss = sum_loss.mean(0)
    print('    Emo, Ene, Fat, Att, Mot, Glo')
    print(' Max Loss:  {}'.format(max_loss.cpu().detach().numpy()))
    print(' Min Loss:  {}'.format(min_loss.cpu().detach().numpy()))
    print(' Mean Loss: {}'.format(mean_loss.cpu().detach().numpy()))
    print(' Average Loss1(total loss): {}'.format(round(float(losses.avg), 4)))
    print(' Average Loss2(mse loss): {}'.format(round(float(sum_loss.mean()), 4)))
    print(' Average accuracy: {}'.format(round(float(accuracies.avg), 3)))
    early_stopping(round(float(losses.avg),4), model)

    return losses.avg, float(sum_loss.mean()), accuracies.avg

if __name__ == '__main__':
   
    main()
