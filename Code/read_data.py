import numpy as np
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
import cv2
import pdb
import random


random.seed(4603)
np.random.seed(4603)
def load_imgs_total_frame(video_root, video_list, data_time=1):
    imgs_first_dict = {}
    imgs_first = []
    n_ind = 0
    video_names = []
    index = []
    with open(video_list, 'r') as imf:
        imf = imf.readlines()

        for id, line in enumerate(imf):

            video_label = line.strip().split()

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
            # print(len(imgs_first_dict[video_name]))
            ###  return video frame index  #####
            ed_list = sample_seg_full(imgs_first_dict[video_name])
            for seg_list in ed_list:
                for data_th in range(data_time):
                    imgs_first.append(seg_list)
                    index.append(n_ind)
                    n_ind += 1

        # index = np.concatenate(index, axis=0)
        # video_names.append(video_name)
        ind = np.arange(0,len(index),1)
    return imgs_first, index


def sample_seg_full(orig_list, seg_num=32):
    ed_list = []
    part = int(len(orig_list)) // seg_num
    if part == 0:
        print('less 32')
    else:
        for n in range(int(part)):
            ed_list.append(orig_list[n * seg_num: n * seg_num + seg_num])

    return ed_list


def sample_seg(orig_list, seg_num=100, seg_num2=64):
    s_list = []
    m_list = []
    ed_list = []
    part = int(len(orig_list)) // seg_num
    if part == 0:
        if len(orig_list) < 32:
            print("less 32")
        elif len(orig_list) < seg_num2:
            m_list.append(orig_list)
            for rand_s in m_list:
                # rand_list = random.sample(rand_s, 32)
                # ed_list.append(rand_list)
                rand_num = random.sample(range(len(rand_s)), 32)
                rand_num.sort()
                ed_list.append([rand_s[i] for i in rand_num])
        else:
            s_list.append(orig_list)
            for seg in s_list:
                rand_n = np.random.randint(len(seg) - seg_num2 + 1, size=1)
                m_list.append(seg[int(rand_n):int(rand_n) + seg_num2])
            for rand_s in m_list:
                # rand_list = random.sample(rand_s, 32)
                # ed_list.append(rand_list)
                rand_num = random.sample(range(len(rand_s)), 32)
                rand_num.sort()
                ed_list.append([rand_s[i] for i in rand_num])

        return ed_list

    for n in range(int(part)):
        s_list.append(orig_list[n * seg_num: n * seg_num + seg_num])
    for seg in s_list:
        rand_n = np.random.randint(seg_num - seg_num2 + 1, size=1)
        m_list.append(seg[int(rand_n):int(rand_n) + seg_num2])
    for rand_s in m_list:
        # rand_list = random.sample(rand_s, 32)
        # ed_list.append(rand_list)
        rand_num = random.sample(range(64), 32)
        rand_num.sort()
        ed_list.append([rand_s[i] for i in rand_num])

    return ed_list


def sample_seg_impro(orig_list, seg_num=100, seg_num2=64, seg_num3=32, dataaug=2):
    s_list = []
    m_list = []
    ed_list = []
    part = int(len(orig_list)) // seg_num
    if part == 0:
        if len(orig_list) < seg_num3:
            pass
            # print("less 32")
        elif len(orig_list) < seg_num2:
            m_list.append(orig_list)
            for rand_s in m_list:
                # rand_list = random.sample(rand_s, 32)
                # ed_list.append(rand_list)
                rand_num = random.sample(range(len(rand_s)), seg_num3)
                rand_num.sort()
                ed_list.append([rand_s[i] for i in rand_num])
        else:
            s_list.append(orig_list)
            for seg in s_list:
                for i in range(dataaug):
                    rand_n = np.random.randint(len(seg) - seg_num2 + 1, size=1)
                    m_list.append(seg[int(rand_n):int(rand_n) + seg_num2])
            for rand_s in m_list:
                # rand_list = random.sample(rand_s, 32)
                # ed_list.append(rand_list)
                rand_num = random.sample(range(len(rand_s)), seg_num3)
                rand_num.sort()
                ed_list.append([rand_s[i] for i in rand_num])

        return ed_list

    for n in range(int(part)):
        s_list.append(orig_list[n * seg_num: n * seg_num + seg_num])
    for seg in s_list:
        for i in range(dataaug):
            rand_n = np.random.randint(seg_num - seg_num2 + 1, size=1)
            m_list.append(seg[int(rand_n):int(rand_n) + seg_num2])
    for rand_s in m_list:
        # rand_list = random.sample(rand_s, 32)
        # ed_list.append(rand_list)
        rand_num = random.sample(range(seg_num2), seg_num3)
        rand_num.sort()
        ed_list.append([rand_s[i] for i in rand_num])

    return ed_list


class FrameAttentionDataSet(data.Dataset):
    '''
    This dataset return entire frames for a video. this means that the number of return for each time is different.
    sample_rate: num_of_image per second
    '''

    def __init__(self, video_root, video_list, transform=None, sample_rate=None, transformVideoAug=None,
                 transformVideoAug_com=None, data_time=1, extremun=False, is_random=False):
        self.imgs_first_dict, self.indexes = load_imgs_total_frame(video_root, video_list,
                                                                   data_time=data_time)
        self.transform = transform
        self.sample_rate = sample_rate
        self.transformVideoAug = transformVideoAug
        self.transformVideoAug_com = transformVideoAug_com
        self.data_time = data_time

    def __getitem__(self, index):
        image_label = self.imgs_first_dict[self.indexes[index]]
        image_list = []
        target_list = []

        for item, fatigue in image_label:
            img = Image.open(item).convert("RGB")
            # img_cv2 = cv2.imread(item)
            # img_resize = image_preporcess(img_cv2)
            # img_from_cv2 = Image.fromarray(np.uint8(img_resize[:, :, ::-1]))
            # img = img_from_cv2
            # if self.transform is not None:
            #     img_ = self.transform(img)
            # else:
            #     img_ = img
            img_ = img
            image_list.append(img_)

            sample = {'fatigue': fatigue}

            target_list.append(sample)

        if self.data_time == 1:
            if self.transformVideoAug is not None:
                image_list = self.transformVideoAug(image_list)
            elif self.transformVideoAug_com is not None:
                image_list = self.transformVideoAug_com(image_list)
        else:
            if index % 2 == 0:
                if self.transformVideoAug is not None:
                    image_list = self.transformVideoAug(image_list)
            else:
                if self.transformVideoAug_com is not None:
                    image_list = self.transformVideoAug_com(image_list)


        if self.transform is not None:
            image_list = [self.transform(image) for image in image_list]
        #print(len(image_list))
        #print(type(image_list[0]))
        target_list = target_list[:len(image_list)]
        #print(len(image_list))
        image_list = np.stack(image_list, axis=0)
        # target_list = np.stack(target_list, axis=0)

        return image_list, target_list

    def __len__(self):
        return len(self.indexes)

def load_imgs_numpy(video_root, video_list, data_doubled=False, Nclass=10):
    imgs_first_dict = {}
    imgs_first = []
    n_ind = 0

    index = []
    with open(video_list, 'r') as imf:
        imf = imf.readlines()

        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name, emotion, energy, fatigue, attention, motivate, Global_Status = video_label

            emotion = (np.float32(emotion) - 1) / 9.0
            energy = (np.float32(energy) - 1) / 99.
            fatigue = (np.float32(fatigue) - 1) / 4.0
            attention = (np.float32(attention) - 1) / 4.0
            motivate = (np.float32(motivate) - 1) / 4.0
            Global_Status = (np.float32(Global_Status) - 1) / 4.0

            # if len(video_name.split('.')[0])==2:
            #     video_path = os.path.join(video_root, video_name.split('.')[0])  # video_path is the path of each video
            # ###  for sampling triple imgs in the single video_path  ####
            # else:
            video_path = os.path.join(video_root, video_name.replace(".mp4", ""))
            img_lists = os.listdir(video_path)
            img_lists.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
            imgs_first_dict[video_name] = []
            for frame in img_lists:
                imgs_first_dict[video_name].append(
                    os.path.join(video_path, frame))
            # print(len(imgs_first_dict[video_name]))
            ###  return video frame index  #####

            if fatigue == 1 :
                fatigue = Nclass
            else:

                fatigue = int(fatigue * Nclass + 1)
            ed_list = sample_seg(imgs_first_dict[video_name])
            for seg_list in ed_list:
                if data_doubled == True:
                    imgs_first.append(seg_list)
                    index.append(n_ind)
                    n_ind += 1
                    imgs_first.append(seg_list)
                    index.append(n_ind)
                    n_ind += 1
                else:
                    imgs_first.append(seg_list)
                    index.append(fatigue)
                    n_ind += 1

        # index = np.concatenate(index, axis=0)
        # video_names.append(video_name)
        ind = np.arange(0,len(index),1)
    return imgs_first, index
if __name__ == "__main__":
    arg_rootTrain = r'F:\R_data\smallsample'
    arg_listTrain = r'../Data/DataS_Train.txt'
    arg_rooteval = r'F:\R_data\smallsample'
    arg_listeval = r'../Data/DataS_Eval.txt'
    # sample_seg()
    a,b = load_imgs_numpy(arg_rootTrain,arg_listTrain)
    # a, b = load_imgs_total_frame(arg_rootTrain, arg_listTrain)
    train_dataset = FrameAttentionDataSet(
        video_root=arg_rootTrain,
        video_list=arg_listTrain,
        # transformVideoAug=random_transformVideoAug(),
        # transform=transforms.Compose([transforms.ToTensor(),
        #                               transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                                    std=(0.5, 0.5, 0.5))]),
        sample_rate=3.5
    )
    pass
    # pass