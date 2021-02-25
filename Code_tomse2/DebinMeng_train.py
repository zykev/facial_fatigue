import numpy as np
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
import cv2
import pdb


try:
    import cPickle as pickle
except:
    import pickle


def load_BIEMSEEAI_Video(video_root, video_list):
    imgs_first = list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name, emotion, energy, fatigue, attention, motivate, Global_Status = video_label

            emotion = (np.float32(emotion) - 1) / 9.0
            energy = (np.float32(energy) - 1) / 99.
            fatigue = (np.float32(fatigue) - 1) / 4.0
            attention = (np.float32(attention) - 1) / 4.0
            motivate = (np.float32(motivate) - 1) / 4.0
            Global_Status = (np.float32(Global_Status) - 1) / 4.0

            video_path = os.path.join(video_root, video_name.split('.')[0])
            img_lists = os.listdir(video_path)
            img_lists.sort()
            img_count = len(img_lists)

            for frame in img_lists:
                imgs_first.append(
                    (os.path.join(video_path, frame), emotion, energy, fatigue, attention, motivate, Global_Status))

            index.append(np.ones(img_count) * id)

        index = np.concatenate(index, axis=0)
    return imgs_first, index


def load_imgs_total_frame(video_root, video_list):
    imgs_first_dict = {}
    video_names = []
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
                    (os.path.join(video_path, frame), emotion, energy, fatigue, attention, motivate, Global_Status))
            # print(len(imgs_first_dict[video_name]))
            ###  return video frame index  #####
            video_names.append(video_name)

    return imgs_first_dict, video_names


'''=============== ========================    Code for FrameMean   ================================================='''


class BIEMSEEAIVideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, transform=None):
        self.imgs_first, self.index = load_BIEMSEEAI_Video(video_root, video_list)

        self.transform = transform

    def __getitem__(self, index):
        path_first, emotion, energy, fatigue, attention, motivate, Global_Status = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)
        sample = {'emotion': emotion, 'energy': energy, 'fatigue': fatigue, 'attention': attention,
                  'motivate': motivate, 'Global_Status': Global_Status}
        return img_first, sample, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


'''=======================================    Code for FrameAttention   ============================================='''


def image_preporcess(image, target_size=(224,224)):

    # resize 尺寸
    ih, iw = target_size
    # 原始图片尺寸
    h,  w, _ = image.shape

    # 计算缩放后图片尺�?
    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    # 创建一张画布，画布的尺寸就是目标尺�?
    # fill_value=120为灰色画�?
    image_paded = np.full(shape=[ih, iw, 3], fill_value=0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2

    # 将缩放后的图片放在画布中�?
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # 归一化处�?
    # image_paded = image_paded / 255.

    return image_paded


class FrameAttentionDataSet(data.Dataset):
    '''
    This dataset return entire frames for a video. this means that the number of return for each time is different.
    sample_rate: num_of_image per second
    '''

    def __init__(self, video_root, video_list, transform=None, sample_rate=None, transformVideoAug=None):
        self.imgs_first_dict, self.video_names = load_imgs_total_frame(video_root, video_list)
        self.transform = transform
        self.sample_rate = sample_rate
        self.transformVideoAug = transformVideoAug

    def __getitem__(self, index):
        image_label = self.imgs_first_dict[self.video_names[index]]
        image_list = []
        target_list = []

        for item, emotion, energy, fatigue, attention, motivate, Global_Status in image_label:
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

            sample = {'emotion': emotion, 'energy': energy, 'fatigue': fatigue, 'attention': attention,
                      'motivate': motivate, 'Global_Status': Global_Status}

            target_list.append(sample)

        if self.transformVideoAug is not None:
            image_list = self.transformVideoAug(image_list)
        if self.transform is not None:
            image_list = [self.transform(image) for image in image_list]
        print(len(image_list))
        print(type(image_list[0]))
        image_list = Sample_Function(image_list, self.sample_rate)
        target_list = target_list[:len(image_list)]
        print(len(image_list))
        image_list = np.stack(image_list, axis=0)
        # target_list = np.stack(target_list, axis=0)

        return image_list, target_list

    def __len__(self):
        return len(self.video_names)


def Sample_Function(frame_list, sample_rate):
    ''' employment sample list by '''
    # print(len(frame_list))
    if sample_rate:
        img_count = len(frame_list)  # number of frames in video
        num_segment = round(img_count / 25. * sample_rate)
        try:
            num_per_part = int(img_count) // num_segment
        except:

            print(frame_list)
            print(sample_rate)

        imgs_item = []
        if num_segment < 32:
            num_segment = 32
            num_per_part = int(img_count) // 32
        elif num_segment > 190:
            num_segment = 190
        # print(num_per_part)
        if num_per_part == 0:
            print("error")
            pass
            # index_s = list(range(int(img_count))) * ((num_segment // int(img_count)) + 1)
            # index_s.sort()
            #
            # split_point = list(range(num_segment))
            # for item in split_point:
            #     imgs_item.append(list(frame_list.keys())[index_s[item]])
            ''' Just for sentence: index.append(np.ones(num_per_part) * id)  # id: 0 : 379 '''

        else:
            index_s = []
            for item in range(num_segment + 1):
                index_s.append(item * num_per_part)

            index_head = index_s[:-1]
            index_tail = index_s[1:];
            index_tail[-1] = img_count

            # for i_group in range(num_per_part):
            for item in range(num_segment):
                s_point = torch.randint(index_head[item], index_tail[item], (1,)).item()
                imgs_item.append(frame_list[s_point])

        return imgs_item
    else:
        return frame_list


if __name__ == "__main__":
    item = r"C:\Users\ASUS\Desktop\target_size_img_1.jpg"
    img_cv2 = cv2.imread(item)
    img_resize = image_preporcess(img_cv2)
    img_from_cv2 = Image.fromarray(np.uint8(img_resize[:, :, ::-1]))
    pass