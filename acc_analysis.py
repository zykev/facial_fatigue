import os
import time
from PIL import Image
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import torch
import torchvision.transforms as transforms
from vidaug import augment as aug
from Code_tomse2 import Model_Parts, util

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=r'/DATA_VD/mood_face/Data-S375-align', type=str, metavar='N',
                        help='the directory of videos need to be predicted')
    parser.add_argument('--label_list', default=r'./Data/acc_test.txt', type=str,
                        help='the list of video names need to be predicted')
    parser.add_argument('--save_dir', default=r'./Data', type=str, metavar='N',
                        help='the directory where preditions need to be predicted')
    parser.add_argument('--model_dir', default=r'./model/03-23_19-37_lr1e-05wd0.001/model/epoch29_loss_0.0584_acc_0.982', type=str, metavar='N',
                        help='the directory where preditions need to be predicted')
    parser.add_argument('--num_classes', default=2, type=int, help='predicted class')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--loss_alpha', default=0.1, type=float,
                        help='adjust loss for crossentrophy')
    args = parser.parse_args()

    return args

# load model

def LoadParameter(_structure, _parameterDir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(_parameterDir, map_location=torch.device(device))
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    return _structure

# load data
def load_imgs_total_frame(video_root, video_list):
    imgs_first_dict = {}
    imgs_first = []
    video_names = []
    with open(video_list, 'r') as imf:
        imf = imf.readlines()

        for id, line in enumerate(imf):

            video_label = line.strip().split(' ')

            video_name, fatigue = video_label
            fatigue = (np.float32(fatigue) - 1) / 4.0

            if video_name.split('.')[-1] == 'mp4':
                video_path = os.path.join(video_root, video_name.replace(".mp4", ""))
            elif video_name.split('.')[-1] == 'mov':
                video_path = os.path.join(video_root, video_name.replace(".mov", ""))
            else:
                video_path = os.path.join(video_root, video_name)

            img_lists = os.listdir(video_path)
            img_lists.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
            imgs_first_dict[video_name] = []
            for frame in img_lists:
                imgs_first_dict[video_name].append(
                    (os.path.join(video_path, frame), fatigue))
            ###  return video frame index  #####
            ed_list = sample_seg_full(imgs_first_dict[video_name])
            imgs_first.append(ed_list)
            video_names.append(video_name)

    return imgs_first, video_names

def sample_seg_full(orig_list, seg_num=32):
    ed_list = []
    part = int(len(orig_list)) // seg_num
    if part == 0:
        print('less 32')
    else:
        for n in range(int(part)):
            ed_list.append(orig_list[n * seg_num: n * seg_num + seg_num])

    return ed_list

# dataloader
class PredDataSet(torch.utils.data.Dataset):
    '''
    This dataset return entire frames for a video. this means that the number of return for each time is different.
    sample_rate: num_of_image per second
    '''

    def __init__(self, imgs_dict, transform=None, transformVideoAug=None):
        self.imgs_first_dict = imgs_dict
        self.transform = transform
        self.transformVideoAug = transformVideoAug

    def __getitem__(self, index):
        image_label = self.imgs_first_dict[index]

        image_list = []
        # target_list = []
        for item, fatigue in image_label:
            img = Image.open(item).convert("RGB")
            img_ = img
            image_list.append(img_)

            sample = float(fatigue)
            target_list = sample
            # target_list.append(sample)


        if self.transformVideoAug is not None:
            image_list = self.transformVideoAug(image_list)


        if self.transform is not None:
            image_list = [self.transform(image) for image in image_list]

        image_list = torch.stack(image_list, dim=0)
        target_list = [torch.tensor(target_list)]



        return image_list, target_list

    def __len__(self):
        return len(self.imgs_first_dict)

def LoadPredData(imgs_dict, batch_size):

    pred_dataset = PredDataSet(
        imgs_dict=imgs_dict,
        # transformVideoAug=transforms.Compose([aug.Resize([112, 112])]),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))])
    )

    pred_loader = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)


    return pred_loader

# prediction

def predict(data_id, val_loader, model, criterion1, criterion2, loss_alpha, summary_statistics):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = util.AverageMeter()
    data_time = util.AverageMeter()
    accuracies = util.AverageMeter()
    end = time.time()
    mse_list = []
    ce_list = []
    label_list = []
    con_label_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (input_image, sample) in enumerate(val_loader):

            sample = sample[0]
            sample_catego = util.label_to_categorical(sample, args.num_classes)
            sample = sample.to(device)
            sample_catego = sample_catego.to(device)

            input_var = torch.autograd.Variable(input_image.squeeze(0)).permute((0, 2, 1, 3, 4))
            input_var = input_var.to(device)

            outputs = model(input_var)

            summary_statistics.update(sample_catego.data.cpu().numpy(), outputs.data.cpu().numpy())


            fatigue_loss_ce = criterion1(outputs, sample_catego)
            outputs_cont = util.output_tomse(outputs, args.num_classes)
            fatigue_loss_mse = criterion2(outputs_cont, sample)

            acc = util.calculate_accuracy(outputs, sample_catego)
            loss = loss_alpha * fatigue_loss_ce + fatigue_loss_mse
            accuracies.update(acc, input_var.size(0))
            losses.update(loss.item(), input_var.size(0))


            mse_list.append(outputs_cont)
            ce_list.append(outputs)
            label_list.append(sample_catego)
            con_label_list.append(sample)

    # measure prediction time
    data_time.update(time.time() - end)
    # outputs and targets for model
    mses = torch.cat(mse_list)
    ces = torch.cat(ce_list)
    labels = torch.cat(label_list)
    con_labels = torch.cat(con_label_list)

    ''' Compute Loss '''

    print(' Data {} prediction overview=============='.format(data_id))
    print(' Pred Time: {}'.format(round(float(data_time.avg), 3)))
    print(' Loss:      {}'.format(round(float(losses.avg), 4)))
    print(' Avg Acc:   {}'.format(round(float(accuracies.avg), 3)))

    return losses.avg, accuracies.avg, mses, ces, labels, con_labels

class SummaryStatistics(object):
    """Generate train/test summary stats
    Tracks the following metrics:
    - Confusion Matrix
    - Average & Per-class precision
    - Average & Per-class recall
    - Average & Per-class acuuracy
    - Average & Per-class f1-score
    """

    def __init__(self, n_classes=4):
        """Constructor
        Args:
            n_classes (int, optional): Number of output classes. Defaults to 4.
        """
        self.n_classes = n_classes
        self.reset() #confusion matrix

    def reset(self):
        """Reset the stats."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, true_labels, pred_labels):
        """Update the confusion matrix and metrics
        Args:
            true_labels (np.array): Actual labels
            pred_labels (np.array): Predicted labels
        """
        if len(pred_labels.shape) > 1:
            pred_labels = np.argmax(pred_labels, axis=-1) #output the class with largest probability
        conf_matrix = np.bincount(
            self.n_classes * true_labels.astype(int) + pred_labels.astype(int),
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)

        self.confusion_matrix += conf_matrix


    def get_metrics(self):
        """Generate/Retrieve the summary metrics.
        Returns:
            [dict]: All metrics mentioned above.
        """
        conf_matrix = self.confusion_matrix
        precision_per_class = np.nan_to_num(
            np.diag(conf_matrix) / np.sum(conf_matrix, axis=0))
        recall_per_class = np.nan_to_num(
            np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
        acc_per_class = np.nan_to_num(np.diag(conf_matrix) / (np.sum(
            conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)))
        f1_per_class = np.nan_to_num(
            2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class))

        avg_precision = np.nanmean(precision_per_class)
        avg_recall = np.nanmean(recall_per_class)
        avg_acc = np.nanmean(acc_per_class)
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

        result = {
            'conf_matrix':  conf_matrix,
            'stats_per_class':  {
                'class_precision':  precision_per_class,
                'class_recall': recall_per_class,
                'class_accuracy':   acc_per_class,
                'class_f1': f1_per_class
            },
            'avg_stats': {
                'avg_precision':    avg_precision,
                'avg_recall':   avg_recall,
                'avg_accuracy': avg_acc,
                'avg_f1':   avg_f1
            }
        }

        return result


args = get_args()
# dir_model = "./model/03-22_12-11_lr0.001wd0.0001/model/epoch33_loss_1.6888_acc_0.789"
model = Model_Parts.FullModal_VisualFeatureAttention(num_class=2, feature_dim=256, non_local_pos=3,
                                                     first_channel=64)

model = LoadParameter(model, args.model_dir)
model = torch.nn.DataParallel(model)

criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()

summary_statistics = SummaryStatistics(args.num_classes)
ce_results_agg = []
labels_agg = []
imgs_first, video_names = load_imgs_total_frame(args.data_dir, args.label_list)
for data_id, (imgs_dict, video_name) in enumerate(zip(imgs_first, video_names)):
    pred_loader = LoadPredData(imgs_dict, args.batch_size)
    totalloss, acc, mse_results, ce_results, labels, con_labels = predict(data_id, pred_loader, model, criterion1,
                                                                          criterion2, args.loss_alpha, summary_statistics)

    _, pred = ce_results.topk(1, 1, largest=True, sorted=True)
    pred = pred.t().squeeze(0)

    avg_labels = 1 if len(pred[pred == 1]) > len(pred) - len(pred[pred == 1]) else 0
    avg_mse_labels = mse_results.mean().cpu().numpy()
    with open(os.path.join(args.save_dir, "prediction.txt"), "a+") as f:
        f.write(video_name + ' ' + str(totalloss) + ' ' + str(acc) + ' ' +
                str(avg_mse_labels) + ' ' + str(con_labels[0].cpu().numpy()) + ' ' +
                str(avg_labels) + ' ' + str(labels[0].cpu().numpy()) + '\n')

    ce_results_agg.append(ce_results.cpu())
    labels_agg.append(labels.cpu())

result_summary = summary_statistics.get_metrics()
whole_ce_results = torch.cat(ce_results_agg)
whole_labels = torch.cat(labels_agg)
whole_probs = torch.nn.functional.softmax(whole_ce_results, dim=1)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(args.num_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(whole_labels.numpy(), whole_probs[:, i].numpy(), pos_label=i)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(args.num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= args.num_classes
fpr["avg"] = all_fpr
tpr["avg"] = mean_tpr
roc_auc["avg"] = metrics.auc(fpr["avg"], tpr["avg"])

result_summary['roc_auc'] = roc_auc
with open(os.path.join(args.save_dir, 'pred_summary.txt'), 'w') as handle:
    handle.write(str(result_summary))

for key in fpr.keys():
    roc_data = np.vstack((fpr[key], tpr[key])).T
    np.savetxt(os.path.join(args.save_dir, 'roc_data_class{}.txt'.format(key)), roc_data, delimiter=' ')


plt.clf()
plt.figure()
colors = ['green', 'darkorange', 'navy']
for key, color in zip(fpr.keys(), colors):
    plt.plot(fpr[key], tpr[key], color=color,
             lw=2, label='ROC curve of class {} (area = {:.2f})'.format(key, roc_auc[key]))

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.save_dir, 'roc_curve.jpg'))
# plt.show()