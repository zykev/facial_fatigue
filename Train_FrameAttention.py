# encoding: utf-8

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
from Code import load_materials, util, Model_Parts, pytorchtools
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
parser.add_argument('--weight-decay', '--wd', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 0.1)')
parser.add_argument('--pf', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--is_test', default=True, dest='is_test',
                    help='testing when is traing (default: True')
parser.add_argument('--is_pretreat', default=False, dest='is_pretreat',
                    help='pretreating when is traing (default: False')
parser.add_argument('--accumulation_step', default=1, type=int, metavar='M',
                    help='accumulation_step')

parser.add_argument('--first_channel', default=64, type=int,
                    help='number of channel in first convolution layer in resnet (default: 64)')
parser.add_argument('--non_local_pos', default=3, type=int,
                    help='the position to add non_local block')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size (default: 32)')
parser.add_argument('--data_time', default=1, type=int,
                    help='the time of auging data')

parser.add_argument('--arg_rootTrain', default=None, type=str,
                    help='the path of train sample ')
parser.add_argument('--arg_rootEval', default=None, type=str,
                    help='the path of eval sample ')


best_prec1 = 10
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
'''MyNote '''


def abs_double(input, target):
    return abs(input - target)

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self,input,target):
        return abs_double(input, target)

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is continuous value (BS, )
        y = y.view(-1)
        x = x.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1-ccc


def load_model(dir_model):
    MSEcriterion = nn.MSELoss()
    # MSEcriterion = Abs()

    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=1, feature_dim=512, at_type='nonLocal')
    model = Model_Parts.LoadParameter(model, dir_model)

    model1 = torch.nn.DataParallel(Model_Parts.FullModel_Loss(model, MSEcriterion))
    return model1


def get_path(lr, wd):
    save_name = datetime.now().strftime('%m-%d_%H-%M')
    folder = './model/' + save_name + '_' + 'lr'+str(lr)+'wd'+str(wd)
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
    global args, best_prec1
    accumulation_step = args.accumulation_step
    data_time = args.data_time
    print('epochs', args.epochs)
    print('learning rate:', args.lr)
    print('weight decay:', args.weight_decay)
    print('accumulation_step:', accumulation_step)
    print('first_channel', args.first_channel)
    print('non_local_pos', args.non_local_pos)
    print('batch size:', args.batch_size)
    print('data_time', data_time)
    print('is_pretreat:', args.is_pretreat)

    # save model superparameter
    with open(os.path.join(new_folder, "hyperparam.txt"), "a+") as f:
        f.write('epochs' + ' ' + str(args.epochs) + '\n')
        f.write('learning rate' + ' ' + str(args.lr) + '\n')
        f.write('weight decay' + ' ' + str(args.weight_decay) + '\n')
        f.write('accumulation step' + ' ' + str(accumulation_step) + '\n')
        f.write('first channel' + ' ' + str(args.first_channel) + '\n')
        f.write('non local pos' + ' ' + str(args.non_local_pos) + '\n')
        f.write('batch size' + ' ' + str(args.batch_size) + '\n')
        f.write('is pretreat' + ' ' + str(args.is_pretreat) + '\n')

    dir_model = r"./model/epoch51_69.0"

    ''' Load data '''

    if args.arg_rootTrain == None:
        arg_listTrain = r'./Data/label-train.txt'
    else:
        arg_listTrain = args.arg_rootTrain
    arg_rootTrain = r'/home/biai/BIAI/mood/Data-S375-align'

    if args.arg_rootEval == None:
        arg_listeval = r'./Data/label-eval.txt'
    else:
        arg_listeval = args.arg_rootEval
    arg_rooteval = r'/home/biai/BIAI/mood/Data-S375-align'

    train_loader, val_loader = load_materials.LoadVideoAttention(arg_rootTrain, arg_listTrain, arg_rooteval,
                                                                      arg_listeval, data_time=data_time, batch_size=args.batch_size)
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
    MSEcriterion = nn.MSELoss()
    CCCcriterion = CCCLoss()
    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=1, feature_dim=feature_dim, non_local_pos=args.non_local_pos,
                                                         first_channel=first_channel)
    if args.is_pretreat:
        print("pretreat.............")
        model = Model_Parts.LoadParameter(model, dir_model)
       
        
    model = torch.nn.DataParallel(Model_Parts.FullModel_Loss(model, MSEcriterion))
    model = torch.nn.DataParallel(Model_Parts.FullModel_Loss(model, CCCcriterion))

    ''' Loss & Optimizer '''
    criterion = nn.CrossEntropyLoss().to(device)
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
    #print('args.lr', args.lr)

    for epoch in range(args.epochs):
        util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)

        print('...... Beginning Train epoch: {} ......'.format(epoch))
        mseloss = train(train_loader, model, criterion, optimizer, epoch, accumulation_step)
        print('...... Beginning Test epoch: {} ......'.format(epoch))
        if args.is_test :
            mseloss1 = validate(val_loader, model)
        else:
            mseloss1 = mseloss
        is_better = mseloss1 < best_prec1
        
        if epoch % 100 == 99:
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': mseloss1,
            }, path=new_folder)
        
        if is_better:
            print('better model!')
            best_prec1 = min(mseloss1, best_prec1)
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': mseloss1,
            }, path=new_folder)
        else:
            print('Model too bad & not save')
            
        with open(os.path.join(new_folder, "a.txt"), "a+") as f:
            f.write(str(mseloss)+" "+str(mseloss1))
            f.write("\n")
            f.flush()
        if early_stopping.early_stop:
            print("Early stopping")
            break

    util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': mseloss1,
            }, path=new_folder)
            

def train(train_loader, model, criterion, optimizer, epoch, accumulation_step):
    global record_

    losses = util.AverageMeter()
    data_time = util.AverageMeter()
    # switch to train mode
    model.train()
    num_of_data = 0
    end = time.time()
    score_list = []
    for batch_idx, (input_image, sample) in enumerate(train_loader):

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)
        sample = sample[0]
        #emotion = torch.autograd.Variable(sample['emotion']).to(device)
        #energy = torch.autograd.Variable(sample['energy']).to(device)
        fatigue = torch.autograd.Variable(sample['fatigue']).to(device)
        #attention = torch.autograd.Variable(sample['attention']).to(device)
        #motivate = torch.autograd.Variable(sample['motivate']).to(device)
        #Global_Status = torch.autograd.Variable(sample['Global_Status']).to(device)
        input_var = torch.autograd.Variable(input_image)
        input_var = np.transpose(input_var, (0,2,1,3,4))
        # compute output
        loss, compact, frame_outputs, groud_truth = model(input_var, fatigue=fatigue)
        score_list.append(compact)
        ''' multi-task: log_vars weight '''
        # loss = loss.sum()
        loss = loss / accumulation_step
        loss.backward();

        losses.update(loss.item(), input_var.size(0))

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
                .format(
                epoch, batch_idx, len(train_loader),
                data_time=data_time, loss=losses))

    ''' Compute Loss '''
    sum_loss = torch.stack(score_list, dim=0)
    max_loss, _ = sum_loss.max(0)
    min_loss, _ = sum_loss.min(0)
    mean_loss = sum_loss.mean(0)
    print('    Emo, Ene, Fat, Att, Mot, Glo')
    print(' Max Loss:  {}'.format(max_loss.cpu().detach().numpy()))
    print(' Min Loss:  {}'.format(min_loss.cpu().detach().numpy()))
    print(' Mean Loss: {}'.format(mean_loss.cpu().detach().numpy()))
    print(' Average Loss1:             {}'.format(round(float(losses.avg), 4)))
    print(' Average Loss2(class wise): {}'.format(round(float(sum_loss.mean()), 4)))

    return sum_loss.mean()



def validate(val_loader, model):
    global record_
    losses = util.AverageMeter()
    data_time = util.AverageMeter()
    # switch to train mode
    end = time.time()
    score_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_image, sample) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            sample = sample[0]
            #emotion = torch.autograd.Variable(sample['emotion']).to(device)
            #energy = torch.autograd.Variable(sample['energy']).to(device)
            fatigue = torch.autograd.Variable(sample['fatigue']).to(device)
            #attention = torch.autograd.Variable(sample['attention']).to(device)
            #motivate = torch.autograd.Variable(sample['motivate']).to(device)
            #Global_Status = torch.autograd.Variable(sample['Global_Status']).to(device)
            input_var = torch.autograd.Variable(input_image)
            input_var = np.transpose(input_var, (0, 2, 1, 3, 4))
            # compute output
            loss, compact, frame_outputs, groud_truth = model(input_var, fatigue=fatigue)
            score_list.append(compact)
            ''' multi-task: log_vars weight '''
            loss = loss.sum()
            losses.update(loss.item(), input_var.size(0))

            if batch_idx % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format(
                    batch_idx, len(val_loader),
                    data_time=data_time, loss=losses))



    ''' Compute Loss '''
    sum_loss = torch.stack(score_list, dim=0)
    max_loss, _ = sum_loss.max(0)
    min_loss, _ = sum_loss.min(0)
    mean_loss = sum_loss.mean(0)
    print('    Emo, Ene, Fat, Att, Mot, Glo')
    print(' Max Loss:  {}'.format(max_loss.cpu().detach().numpy()))
    print(' Min Loss:  {}'.format(min_loss.cpu().detach().numpy()))
    print(' Mean Loss: {}'.format(mean_loss.cpu().detach().numpy()))
    print(' Average Loss1:             {}'.format(round(float(losses.avg), 4)))
    print(' Average Loss2(class wise): {}'.format(round(float(sum_loss.mean()), 4)))
    early_stopping(round(float(losses.avg),4), model)
    #return sum_loss.mean()
    return sum_loss.mean()

if __name__ == '__main__':
   
    main()
