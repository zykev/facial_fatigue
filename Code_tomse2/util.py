import os
import torch
from torch.nn import Parameter
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from Code_tomse2 import Model_Parts


def save_checkpoint(state):
    if not os.path.exists('./model'):
        os.makedirs('./model')

    save_dir = './model/epoch'+str(state['epoch']) + '_loss_' + str(round(float(state['totalloss']), 4)) + '_acc_' + str(round(float(state['acc']), 3))
    torch.save(state, save_dir, _use_new_zipfile_serialization=False)
    print(save_dir)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, learning_rate, end_epoch):
    if epoch in [round(end_epoch * 0.333), round(end_epoch * 0.666)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2
        learning_rate = learning_rate * 0.2


def adjust_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("no video")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(round(fps)) == ord('q'):
            break
    cap.release()

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size

def calculate_coral_accuracy(outputs, targets):
    cal_probs = np.zeros((outputs.size(0), outputs.size(1) + 2))
    cal_probs[:, 0] = 1
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        probs = probs.detach().numpy()
        cal_probs[:, 1:-1] = probs
        class_probs = -np.diff(cal_probs)
        class_probs = torch.from_numpy(class_probs)
        batch_size = targets.size(0)
        _, pred = class_probs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def label_to_categorical(sample, num_classes): # sample is a tensor
    #sample = np.array([float(i) for i in sample['fatigue']])
    #sample = np.array(sample)
    #sample = sample.reshape(1,)
    bins = np.arange(0, 1, 1 / num_classes)
    bins = np.append(bins, 1.1)
    label_class = pd.cut(np.array(sample), bins, right=False, labels=np.arange(num_classes))
    label_class = label_class.astype(np.int)
    label_class = torch.tensor(label_class, dtype = torch.long)

    return label_class

def output_tomse(outputs, num_classes): # outputs from model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bins = torch.arange(0, 1, 1 / num_classes).to(device)
    avg_step = torch.mean(bins[0:2])
    bins = bins + avg_step
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.sum(outputs * bins, 1)

    return outputs



def pre_load_model(dir_model):
    checkpoint = torch.load(dir_model, map_location=torch.device('cpu'))
    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=6, feature_dim=512, at_type='relation')

    start_epoch = checkpoint['epoch'] + 1
    mseloss_model = checkpoint['prec1']
    key = ['module.model.visual_encoder.conv1.weight', 'module.model.visual_encoder.bn1.weight',
           'module.model.visual_encoder.bn1.bias', 'module.model.visual_encoder.bn1.running_mean',
           'module.model.visual_encoder.bn1.running_var', 'module.model.visual_encoder.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer1.0.conv1.weight', 'module.model.visual_encoder.layer1.0.bn1.weight',
           'module.model.visual_encoder.layer1.0.bn1.bias', 'module.model.visual_encoder.layer1.0.bn1.running_mean',
           'module.model.visual_encoder.layer1.0.bn1.running_var',
           'module.model.visual_encoder.layer1.0.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer1.0.conv2.weight', 'module.model.visual_encoder.layer1.0.bn2.weight',
           'module.model.visual_encoder.layer1.0.bn2.bias', 'module.model.visual_encoder.layer1.0.bn2.running_mean',
           'module.model.visual_encoder.layer1.0.bn2.running_var',
           'module.model.visual_encoder.layer1.0.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer1.1.conv1.weight', 'module.model.visual_encoder.layer1.1.bn1.weight',
           'module.model.visual_encoder.layer1.1.bn1.bias', 'module.model.visual_encoder.layer1.1.bn1.running_mean',
           'module.model.visual_encoder.layer1.1.bn1.running_var',
           'module.model.visual_encoder.layer1.1.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer1.1.conv2.weight', 'module.model.visual_encoder.layer1.1.bn2.weight',
           'module.model.visual_encoder.layer1.1.bn2.bias', 'module.model.visual_encoder.layer1.1.bn2.running_mean',
           'module.model.visual_encoder.layer1.1.bn2.running_var',
           'module.model.visual_encoder.layer1.1.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer2.0.conv1.weight', 'module.model.visual_encoder.layer2.0.bn1.weight',
           'module.model.visual_encoder.layer2.0.bn1.bias', 'module.model.visual_encoder.layer2.0.bn1.running_mean',
           'module.model.visual_encoder.layer2.0.bn1.running_var',
           'module.model.visual_encoder.layer2.0.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer2.0.conv2.weight', 'module.model.visual_encoder.layer2.0.bn2.weight',
           'module.model.visual_encoder.layer2.0.bn2.bias', 'module.model.visual_encoder.layer2.0.bn2.running_mean',
           'module.model.visual_encoder.layer2.0.bn2.running_var',
           'module.model.visual_encoder.layer2.0.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer2.0.downsample.0.weight',
           'module.model.visual_encoder.layer2.0.downsample.1.weight',
           'module.model.visual_encoder.layer2.0.downsample.1.bias',
           'module.model.visual_encoder.layer2.0.downsample.1.running_mean',
           'module.model.visual_encoder.layer2.0.downsample.1.running_var',
           'module.model.visual_encoder.layer2.0.downsample.1.num_batches_tracked',
           'module.model.visual_encoder.layer2.1.conv1.weight', 'module.model.visual_encoder.layer2.1.bn1.weight',
           'module.model.visual_encoder.layer2.1.bn1.bias', 'module.model.visual_encoder.layer2.1.bn1.running_mean',
           'module.model.visual_encoder.layer2.1.bn1.running_var',
           'module.model.visual_encoder.layer2.1.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer2.1.conv2.weight', 'module.model.visual_encoder.layer2.1.bn2.weight',
           'module.model.visual_encoder.layer2.1.bn2.bias', 'module.model.visual_encoder.layer2.1.bn2.running_mean',
           'module.model.visual_encoder.layer2.1.bn2.running_var',
           'module.model.visual_encoder.layer2.1.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer3.0.conv1.weight', 'module.model.visual_encoder.layer3.0.bn1.weight',
           'module.model.visual_encoder.layer3.0.bn1.bias', 'module.model.visual_encoder.layer3.0.bn1.running_mean',
           'module.model.visual_encoder.layer3.0.bn1.running_var',
           'module.model.visual_encoder.layer3.0.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer3.0.conv2.weight', 'module.model.visual_encoder.layer3.0.bn2.weight',
           'module.model.visual_encoder.layer3.0.bn2.bias', 'module.model.visual_encoder.layer3.0.bn2.running_mean',
           'module.model.visual_encoder.layer3.0.bn2.running_var',
           'module.model.visual_encoder.layer3.0.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer3.0.downsample.0.weight',
           'module.model.visual_encoder.layer3.0.downsample.1.weight',
           'module.model.visual_encoder.layer3.0.downsample.1.bias',
           'module.model.visual_encoder.layer3.0.downsample.1.running_mean',
           'module.model.visual_encoder.layer3.0.downsample.1.running_var',
           'module.model.visual_encoder.layer3.0.downsample.1.num_batches_tracked',
           'module.model.visual_encoder.layer3.1.conv1.weight', 'module.model.visual_encoder.layer3.1.bn1.weight',
           'module.model.visual_encoder.layer3.1.bn1.bias', 'module.model.visual_encoder.layer3.1.bn1.running_mean',
           'module.model.visual_encoder.layer3.1.bn1.running_var',
           'module.model.visual_encoder.layer3.1.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer3.1.conv2.weight', 'module.model.visual_encoder.layer3.1.bn2.weight',
           'module.model.visual_encoder.layer3.1.bn2.bias', 'module.model.visual_encoder.layer3.1.bn2.running_mean',
           'module.model.visual_encoder.layer3.1.bn2.running_var',
           'module.model.visual_encoder.layer3.1.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer4.0.conv1.weight', 'module.model.visual_encoder.layer4.0.bn1.weight',
           'module.model.visual_encoder.layer4.0.bn1.bias', 'module.model.visual_encoder.layer4.0.bn1.running_mean',
           'module.model.visual_encoder.layer4.0.bn1.running_var',
           'module.model.visual_encoder.layer4.0.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer4.0.conv2.weight', 'module.model.visual_encoder.layer4.0.bn2.weight',
           'module.model.visual_encoder.layer4.0.bn2.bias', 'module.model.visual_encoder.layer4.0.bn2.running_mean',
           'module.model.visual_encoder.layer4.0.bn2.running_var',
           'module.model.visual_encoder.layer4.0.bn2.num_batches_tracked',
           'module.model.visual_encoder.layer4.0.downsample.0.weight',
           'module.model.visual_encoder.layer4.0.downsample.1.weight',
           'module.model.visual_encoder.layer4.0.downsample.1.bias',
           'module.model.visual_encoder.layer4.0.downsample.1.running_mean',
           'module.model.visual_encoder.layer4.0.downsample.1.running_var',
           'module.model.visual_encoder.layer4.0.downsample.1.num_batches_tracked',
           'module.model.visual_encoder.layer4.1.conv1.weight', 'module.model.visual_encoder.layer4.1.bn1.weight',
           'module.model.visual_encoder.layer4.1.bn1.bias', 'module.model.visual_encoder.layer4.1.bn1.running_mean',
           'module.model.visual_encoder.layer4.1.bn1.running_var',
           'module.model.visual_encoder.layer4.1.bn1.num_batches_tracked',
           'module.model.visual_encoder.layer4.1.conv2.weight', 'module.model.visual_encoder.layer4.1.bn2.weight',
           'module.model.visual_encoder.layer4.1.bn2.bias', 'module.model.visual_encoder.layer4.1.bn2.running_mean',
           'module.model.visual_encoder.layer4.1.bn2.running_var',
           'module.model.visual_encoder.layer4.1.bn2.num_batches_tracked', 'module.model.visual_encoder.fc.weight',
           'module.model.visual_encoder.fc.bias', 'module.model.vectors_attention.alpha.0.weight',
           'module.model.vectors_attention.alpha.0.bias', 'module.model.vectors_attention.beta.0.weight',
           'module.model.vectors_attention.beta.0.bias', 'module.model.liner.weight', 'module.model.liner.bias', ]
    new_key = ['visual_encoder.conv1.weight', 'visual_encoder.bn1.weight', 'visual_encoder.bn1.bias',
               'visual_encoder.bn1.running_mean', 'visual_encoder.bn1.running_var',
               'visual_encoder.bn1.num_batches_tracked', 'visual_encoder.layer1.0.conv1.weight',
               'visual_encoder.layer1.0.bn1.weight', 'visual_encoder.layer1.0.bn1.bias',
               'visual_encoder.layer1.0.bn1.running_mean', 'visual_encoder.layer1.0.bn1.running_var',
               'visual_encoder.layer1.0.bn1.num_batches_tracked', 'visual_encoder.layer1.0.conv2.weight',
               'visual_encoder.layer1.0.bn2.weight', 'visual_encoder.layer1.0.bn2.bias',
               'visual_encoder.layer1.0.bn2.running_mean', 'visual_encoder.layer1.0.bn2.running_var',
               'visual_encoder.layer1.0.bn2.num_batches_tracked', 'visual_encoder.layer1.1.conv1.weight',
               'visual_encoder.layer1.1.bn1.weight', 'visual_encoder.layer1.1.bn1.bias',
               'visual_encoder.layer1.1.bn1.running_mean', 'visual_encoder.layer1.1.bn1.running_var',
               'visual_encoder.layer1.1.bn1.num_batches_tracked', 'visual_encoder.layer1.1.conv2.weight',
               'visual_encoder.layer1.1.bn2.weight', 'visual_encoder.layer1.1.bn2.bias',
               'visual_encoder.layer1.1.bn2.running_mean', 'visual_encoder.layer1.1.bn2.running_var',
               'visual_encoder.layer1.1.bn2.num_batches_tracked', 'visual_encoder.layer2.0.conv1.weight',
               'visual_encoder.layer2.0.bn1.weight', 'visual_encoder.layer2.0.bn1.bias',
               'visual_encoder.layer2.0.bn1.running_mean', 'visual_encoder.layer2.0.bn1.running_var',
               'visual_encoder.layer2.0.bn1.num_batches_tracked', 'visual_encoder.layer2.0.conv2.weight',
               'visual_encoder.layer2.0.bn2.weight', 'visual_encoder.layer2.0.bn2.bias',
               'visual_encoder.layer2.0.bn2.running_mean', 'visual_encoder.layer2.0.bn2.running_var',
               'visual_encoder.layer2.0.bn2.num_batches_tracked', 'visual_encoder.layer2.0.downsample.0.weight',
               'visual_encoder.layer2.0.downsample.1.weight', 'visual_encoder.layer2.0.downsample.1.bias',
               'visual_encoder.layer2.0.downsample.1.running_mean', 'visual_encoder.layer2.0.downsample.1.running_var',
               'visual_encoder.layer2.0.downsample.1.num_batches_tracked', 'visual_encoder.layer2.1.conv1.weight',
               'visual_encoder.layer2.1.bn1.weight', 'visual_encoder.layer2.1.bn1.bias',
               'visual_encoder.layer2.1.bn1.running_mean', 'visual_encoder.layer2.1.bn1.running_var',
               'visual_encoder.layer2.1.bn1.num_batches_tracked', 'visual_encoder.layer2.1.conv2.weight',
               'visual_encoder.layer2.1.bn2.weight', 'visual_encoder.layer2.1.bn2.bias',
               'visual_encoder.layer2.1.bn2.running_mean', 'visual_encoder.layer2.1.bn2.running_var',
               'visual_encoder.layer2.1.bn2.num_batches_tracked', 'visual_encoder.layer3.0.conv1.weight',
               'visual_encoder.layer3.0.bn1.weight', 'visual_encoder.layer3.0.bn1.bias',
               'visual_encoder.layer3.0.bn1.running_mean', 'visual_encoder.layer3.0.bn1.running_var',
               'visual_encoder.layer3.0.bn1.num_batches_tracked', 'visual_encoder.layer3.0.conv2.weight',
               'visual_encoder.layer3.0.bn2.weight', 'visual_encoder.layer3.0.bn2.bias',
               'visual_encoder.layer3.0.bn2.running_mean', 'visual_encoder.layer3.0.bn2.running_var',
               'visual_encoder.layer3.0.bn2.num_batches_tracked', 'visual_encoder.layer3.0.downsample.0.weight',
               'visual_encoder.layer3.0.downsample.1.weight', 'visual_encoder.layer3.0.downsample.1.bias',
               'visual_encoder.layer3.0.downsample.1.running_mean', 'visual_encoder.layer3.0.downsample.1.running_var',
               'visual_encoder.layer3.0.downsample.1.num_batches_tracked', 'visual_encoder.layer3.1.conv1.weight',
               'visual_encoder.layer3.1.bn1.weight', 'visual_encoder.layer3.1.bn1.bias',
               'visual_encoder.layer3.1.bn1.running_mean', 'visual_encoder.layer3.1.bn1.running_var',
               'visual_encoder.layer3.1.bn1.num_batches_tracked', 'visual_encoder.layer3.1.conv2.weight',
               'visual_encoder.layer3.1.bn2.weight', 'visual_encoder.layer3.1.bn2.bias',
               'visual_encoder.layer3.1.bn2.running_mean', 'visual_encoder.layer3.1.bn2.running_var',
               'visual_encoder.layer3.1.bn2.num_batches_tracked', 'visual_encoder.layer4.0.conv1.weight',
               'visual_encoder.layer4.0.bn1.weight', 'visual_encoder.layer4.0.bn1.bias',
               'visual_encoder.layer4.0.bn1.running_mean', 'visual_encoder.layer4.0.bn1.running_var',
               'visual_encoder.layer4.0.bn1.num_batches_tracked', 'visual_encoder.layer4.0.conv2.weight',
               'visual_encoder.layer4.0.bn2.weight', 'visual_encoder.layer4.0.bn2.bias',
               'visual_encoder.layer4.0.bn2.running_mean', 'visual_encoder.layer4.0.bn2.running_var',
               'visual_encoder.layer4.0.bn2.num_batches_tracked', 'visual_encoder.layer4.0.downsample.0.weight',
               'visual_encoder.layer4.0.downsample.1.weight', 'visual_encoder.layer4.0.downsample.1.bias',
               'visual_encoder.layer4.0.downsample.1.running_mean', 'visual_encoder.layer4.0.downsample.1.running_var',
               'visual_encoder.layer4.0.downsample.1.num_batches_tracked', 'visual_encoder.layer4.1.conv1.weight',
               'visual_encoder.layer4.1.bn1.weight', 'visual_encoder.layer4.1.bn1.bias',
               'visual_encoder.layer4.1.bn1.running_mean', 'visual_encoder.layer4.1.bn1.running_var',
               'visual_encoder.layer4.1.bn1.num_batches_tracked', 'visual_encoder.layer4.1.conv2.weight',
               'visual_encoder.layer4.1.bn2.weight', 'visual_encoder.layer4.1.bn2.bias',
               'visual_encoder.layer4.1.bn2.running_mean', 'visual_encoder.layer4.1.bn2.running_var',
               'visual_encoder.layer4.1.bn2.num_batches_tracked', 'visual_encoder.fc.weight', 'visual_encoder.fc.bias',
               'vectors_attention.alpha.0.weight', 'vectors_attention.alpha.0.bias', 'vectors_attention.beta.0.weight',
               'vectors_attention.beta.0.bias', 'liner.weight', 'liner.bias']
    weight_chgnm = dict()
    for i in range(len(new_key)):
        weight_chgnm[new_key[i]] = checkpoint['state_dict'][key[i]]

    model.load_state_dict(weight_chgnm)

    return model


def write_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    video_writer = cv2.VideoWriter('outputVideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("no video")
            break
        frame = spin_video(frame, "left")
        cv2.imshow('frame', frame)
        video_writer.write(frame)
        if cv2.waitKey(round(fps)) == ord('q'):
            break
    cap.release()

def spin_video(frame, selection):
    hight, width = frame.shape[:2]
    print(hight, width)
    new_frame = []
    if selection == "left":
        for col in range(width):
            new_frame.append(frame[:, width - col - 1, :])
        new_frame = np.array(new_frame)
    if selection == "right":
        for row in range(hight):
            if row == 0:
                new_frame = np.expand_dims(frame[hight - row - 1, :, :], axis=1)
            # np.c_[new_frame, frame[hight - row - 1, :, :]]
            # np.insert(new_frame, values=frame[hight - row - 1, :, :], axis=1)
            elif row > 0:
                new_frame = np.concatenate((new_frame, np.expand_dims(frame[hight - row - 1, :, :], axis=1)), axis=1)

    return new_frame


def video2frame(Parent_dir, Frame_dir):
    for root, dirs, files in os.walk(Parent_dir):
        for name in files:

            tail = name.split('.')[-1]
            # pdb.set_trace()
            if tail in ['mp4', 'webm', 'avi', 'mov']:

                inputVideo = os.path.join(root, name)
                frameOutput = inputVideo.replace(Parent_dir, Frame_dir).replace('.'+tail,'')
                if not os.path.exists(frameOutput):
                    os.makedirs(frameOutput)

                cap = cv2.VideoCapture(root + '/' + name)
                fps = cap.get(cv2.CAP_PROP_FPS);
                fps = int(fps)
                get_image_rate = round(fps / 5);
                c = 0
                face_num = 0;
                rotate_angle = 0;
                find_face_times = 0;
                # pro_file = __file__;
                # pro_dir = pro_file[:pro_file.rfind("/")];
                classfier = cv2.CascadeClassifier(".." + "/haarcascades/haarcascade_frontalface_alt.xml");
                selection = 0
                minSize = (80, 80);

                while(1):
                    success, frame = cap.read()
                    if success:
                        c = c + 1
                        # if (c % (get_image_rate)) != 0:
                        #     continue
                        if (find_face_times <= 5):
                            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4,
                                                                   minSize=minSize)
                            if len(faceRects) > 0:
                                face_num = len(faceRects);
                                find_face_times = find_face_times + 1;

                            else:
                                (h, w) = grey.shape[:2]
                                center = (w // 2, h // 2)
                                M = cv2.getRotationMatrix2D(center, 90, 1.0)
                                rotated = cv2.warpAffine(grey, M, (w, h))
                                faceRects = classfier.detectMultiScale(rotated, scaleFactor=1.09, minNeighbors=4,
                                                                       minSize=minSize)
                                if len(faceRects) > 0:
                                    selection = "left"
                                    face_num = len(faceRects)
                                    rotate_angle = 90;
                                    find_face_times = find_face_times + 1;
                                else:
                                    M = cv2.getRotationMatrix2D(center, -90, 1.0)
                                    rotated = cv2.warpAffine(grey, M, (w, h))
                                    find_face_times = find_face_times + 1;
                                    faceRects = classfier.detectMultiScale(rotated, scaleFactor=1.09,
                                                                           minNeighbors=4, minSize=minSize)
                                    if len(faceRects) > 0:
                                        selection = "right"
                                        face_num = len(faceRects)
                                        rotate_angle = -90;

                        if selection in ["left", "right"]:
                            frame = spin_video(frame, selection)
                        cv2.imwrite(frameOutput + "/" + str(c) + '.jpg', frame)
                        # if (c % (get_image_rate)) == 0:
                        #     if face_num > 0 and rotate_angle != 0 and 'M' in vars():
                        #         rotated_final = cv2.warpAffine(frame, M, (w, h))
                        #         cv2.imwrite(frameOutput+ "/" + str(c) + '.jpg', rotated_final)
                        #     else:
                        #

                        rotate_angle = 0;
                    else:
                        break
                cap.release()
                if face_num == 0 or find_face_times < 5:
                    print("result:",name + " no face");
                    print(root + '/' + name)
                    # return '';


if __name__ == "__main__":
    # import time
    # time1 = time.time()
    # path = r"F:\Documents\Data\Data_test\IMG_23.mp4"
    # # adjust_video(path)
    #
    # write_video(path)

    # path_frame = r"F:\Documents\Data\Data_test\a\1.jpg"
    # frame = cv2.imread(path_frame)
    # a = spin_video(frame, "right")
    #
    # time2 = time.time()
    # print(time2 - time1)
    # pass
    #
    # Parent_dir = '../Data-2_Face'
    # Frame_dir = '../Data-2_Face'

    Parent_dir = r'F:\Documents\Data\Data_test'
    Frame_dir = r'F:\Documents\Data\Data_test'

    video2frame(Parent_dir, Frame_dir)