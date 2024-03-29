# coding=utf-8

from __future__ import print_function
import torch.utils.data
import torchvision.transforms as transforms
import torch
from Code import read_data
from vidaug import augment as aug



def LoadParameter(_structure, _parameterDir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(_parameterDir, map_location=torch.device(device))
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        model_state_dict[key.replace('module.model', '')] = pretrained_state_dict[key]
        # if '.fc.' in key:
        #     pass
        # else:
        #     model_state_dict[key.replace('module.model', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    return _structure


def random_transformVideoAug_more():


    sometimes = lambda aug_prob: aug.Sometimes(0.5, aug_prob)  # Used to apply augmentor with 50% probability

    sub1 = transforms.Compose([aug.Resize([112, 112]),
                               sometimes(aug.HorizontalFlip()),
                               sometimes(aug.GaussianBlur([0.1, 0.6])),
                               sometimes(aug.Add(-50, 20)),
                               sometimes(aug.EnhanceColor([0.8, 1.2], 'color'))
                               ])

    sub2 = transforms.Compose([aug.Resize([112, 112]),
                               sometimes(aug.HorizontalFlip()),
                               sometimes(aug.GaussianBlur([0.1, 0.6])),
                               sometimes(aug.Add(30, 20)),
                               sometimes(aug.EnhanceColor([0.8, 1.2], 'contrast'))
                               ])


    sub = [sub1,sub2]

    tansf = transforms.RandomChoice(sub)
    return tansf



def LoadVideoAttention(root_train, arg_train_list, root_eval, arg_test_list, data_time, batch_size):
    train_dataset = read_data.FrameAttentionDataSet(
        video_root=root_train,
        video_list=arg_train_list,
        transformVideoAug=random_transformVideoAug_more(),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))]),
        data_time=data_time,
        sample_rate=3.5
    )
    val_dataset = read_data.FrameAttentionDataSet(
        video_root=root_eval,
        video_list=arg_test_list,
        transformVideoAug=transforms.Compose([aug.Resize([112, 112])]),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))]),
        data_time=data_time,
        sample_rate=3.5)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=True)

    return train_loader, val_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    arg_rootTrain = r'F:\R_data\DataSseg_Face\smallsample1'
    # arg_listTrain = '../Data/DataS-Train.txt'
    arg_listTrain = r'C:\Users\ASUS\Desktop\新建文件夹\DataS_Train.txt'
    arg_rooteval = r'F:\R_data\DataSseg_Face\smallsample1'
    # arg_listeval = '../Data/DataS-Eval.txt'
    arg_listeval = r'C:\Users\ASUS\Desktop\新建文件夹\DataS_Eval.txt'
    train_loader, val_loader = LoadVideoAttention(arg_rootTrain, arg_listTrain, arg_rooteval, arg_listeval)
    for i, (va, index) in enumerate(train_loader):
        for j in range(len(index)):
            # val = np.transpose(va,(0,1,3,4,2))
            plt.imshow(va[0, j,0, :, :])
            plt.pause(1)
        pass