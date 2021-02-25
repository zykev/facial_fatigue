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
from Code_tomse import load_materials, util, Model_Parts, pytorchtools
import time
from datetime import datetime
import pdb

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-4)')
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
parser.add_argument('--accumulation_step', default=16, type=int, metavar='M',
                    help='accumulation_step')
parser.add_argument('--loss_alpha', default=1/50, type=float,
                    help='adjust loss for crossentrophy')


best_prec_total1 = 10
best_prec_mse1 = 10
save_name = datetime.now().strftime('%m-%d_%H-%M') + '.txt'
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
'''MyNote '''


def abs_double(input, target):
    return abs(input - target)

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self,input,target):
        return abs_double(input, target)

class CoralLoss(nn.Module):
    def __init__(self,weight=1):
        super(CoralLoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, targets):

        log_prob = (nn.functional.logsigmoid(outputs) * targets + (nn.functional.logsigmoid(outputs) - targets) * (1 - targets)) * self.weight
        loss = - torch.sum(log_prob, dim = 1)
        loss = torch.mean(loss)

        return loss

def coral_loss(logits, levels, imp=1):
    val = -torch.sum((nn.functional.logsigmoid(logits) * levels + (nn.functional.logsigmoid(logits) - logits) * (1 - levels)) * imp, dim = 1)
    return torch.mean(val)


def load_model(dir_model):
    MSEcriterion = nn.MSELoss()


    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=1, feature_dim=512, at_type='nonLocal')
    model = Model_Parts.LoadParameter(model, dir_model)

    model1 = torch.nn.DataParallel(Model_Parts.FullModel_Loss(model, MSEcriterion))
    return model1

early_stopping = pytorchtools.EarlyStopping(patience=20)

def main():
    global args, best_prec_total1, best_prec_mse1, device
    accumulation_step = args.accumulation_step
    print('learning rate:', args.lr)
    print('weight decay:', args.weight_decay)
    print('accumulation_step:', accumulation_step)
    print('loss alpha:', args.loss_alpha)
    dir_model = r"./model/epoch47_0.0653"

    ''' Load data '''

    # arg_rootTrain = r'/home/biai/BIAI/mood/DataS_Face/'
    # arg_listTrain = r'./Data/DataS_Train.txt'
    # arg_rooteval = r'/home/biai/BIAI/mood/DataS_Face/'
    # arg_listeval = r'./Data/DataS_Eval.txt'

    arg_rootTrain = r'/home/biai/BIAI/mood/Data-S235/'
    arg_listTrain = r'/home/biai/BIAI/mood/onlyfat_selfa3DS_class_tomse/Data/Data-fatigue-Train.txt'
    arg_rooteval = r'/home/biai/BIAI/mood/Data-S235/'
    arg_listeval = r'/home/biai/BIAI/mood/onlyfat_selfa3DS_class_tomse/Data/Data-fatigue-Eval.txt'

    train_loader, val_loader = load_materials.LoadVideoAttention(arg_rootTrain, arg_listTrain, arg_rooteval,
                                                                      arg_listeval)
    ''' Eval '''
    print('args.evaluate', args.evaluate)
    loopTest = False
    if args.evaluate:
        if loopTest == True:
            dir_path = r"F:\Documents\biai-release_test\model"
            for mod in os.listdir(dir_path):
                print('-'*10,mod)
                dir_model = dir_path + "/" + mod
                model = load_model(dir_model)
                validate(val_loader, model)
        else:

            model = load_model(dir_model)
            validate(val_loader, model)

        return

    ''' Load model '''
    # MSEcriterion = nn.MSELoss()
    criterion1 = CoralLoss(weight=1)
    criterion2 = nn.MSELoss()
    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=10, feature_dim=256, at_type='nonLocal')
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
            })
        
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
            })
        else:
            print('Model too bad & not save')
            
        with open(save_name, "a+") as f:
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
            })
            

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
    for batch_idx, (input_image, sample, sample_bi) in enumerate(train_loader):
        # measure data loading time
        # print(len(sample))
        # print(sample[0])
        data_time.update(time.time() - end)
        #sample = sample[0]
        sample_catego = util.label_to_categorical(sample, 10)
        sample = sample.to(device)
        sample_bi = sample_bi.to(device)
        sample_catego = sample_catego.to(device)

        input_var = torch.autograd.Variable(input_image).permute((0, 2, 1, 3, 4))
        input_var = input_var.to(device)
        #input_var = np.transpose(input_var, (0, 2, 1, 3, 4))
        outputs = model(input_var)
        # emotion = torch.autograd.Variable(sample['emotion']).to(device)
        # energy = torch.autograd.Variable(sample['energy']).to(device)
        # fatigue = torch.autograd.Variable(sample['fatigue']).to(device)
        # attention = torch.autograd.Variable(sample['attention']).to(device)
        # motivate = torch.autograd.Variable(sample['motivate']).to(device)
        # Global_Status = torch.autograd.Variable(sample['Global_Status']).to(device)

        #fatigue_loss = criterion(outputs, torch.Tensor([int(i) for i in sample['fatigue']]).long())
        fatigue_loss_ce = criterion1(outputs, sample_bi)
        outputs_cont, class_probs = util.output_coral_tomse(outputs, 10)
        fatigue_loss_mse = criterion2(outputs_cont, sample)

        #acc = util.calculate_accuracy(outputs, torch.Tensor([int(i) for i in sample['fatigue']]).long())
        acc = util.calculate_coral_accuracy(class_probs, sample_catego)
        
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
        for batch_idx, (input_image, sample, sample_bi) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            #sample = sample[0]
            sample_catego = util.label_to_categorical(sample, 10)
            sample = sample.to(device)
            sample_bi = sample_bi.to(device)
            sample_catego = sample_catego.to(device)

            input_var = torch.autograd.Variable(input_image).permute((0, 2, 1, 3, 4))
            input_var = input_var.to(device)
            #input_var = np.transpose(input_var, (0, 2, 1, 3, 4))
            outputs = model(input_var)

            #input_var = torch.autograd.Variable(input_image)
            #input_var = np.transpose(input_var, (0, 2, 1, 3, 4))
            # compute output
            # loss, compact, frame_outputs, groud_truth = model(input_var, emotion, energy, fatigue, attention, motivate,
            #                                                   Global_Status)

            #fatigue_loss = criterion(outputs, torch.Tensor([int(i) for i in sample['fatigue']]).long())
            #acc = util.calculate_accuracy(outputs, torch.Tensor([int(i) for i in sample['fatigue']]).long())
            fatigue_loss_ce = criterion1(outputs, sample_bi)
            outputs_cont, class_probs = util.output_coral_tomse(outputs, 10)
            fatigue_loss_mse = criterion2(outputs_cont, sample)


            acc = util.calculate_coral_accuracy(class_probs, sample_catego)

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
    #return sum_loss.mean()
    return losses.avg, float(sum_loss.mean()), accuracies.avg

if __name__ == '__main__':
   
    main()
