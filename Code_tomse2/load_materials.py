# coding=utf-8

from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch
from Code_tomse2 import DebinMeng_train, read_data
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


def LoadFrameAttention(root_train, arg_train_list, root_eval, arg_test_list):
    train_dataset = DebinMeng_train.FrameAttentionDataSet(
        video_root=root_train,
        video_list=arg_train_list,
        transform=transforms.Compose([transforms.Resize(240), transforms.RandomCrop(224),
                                      # transforms.ColorJitter(0.03,0.03,0.03,0.03),
                                      transforms.RandomRotation(2),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(3),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))
                                      ]),

        sample_rate=5
    )
    val_dataset = DebinMeng_train.BIEMSEEAIVideoDataset(
        video_root=root_eval,
        video_list=arg_test_list,
        transform=transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))
                                      ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    return train_loader, val_loader

def random_transformVideoAug():

    sometimes = lambda aug_prob: aug.Sometimes(0.5, aug_prob)  # Used to apply augmentor with 100% probability
    sometimes_low = lambda aug_prob: aug.Sometimes(0.2, aug_prob)
    # seq = aug.Sequential([aug.GaussianBlur(2)])
    # seq = aug.Sequential([aug.PiecewiseAffineTransform(1,1,2)])
    # seq = aug.Sequential([aug.ConvertGray()])
    # seq = aug.Sequential([aug.Add(-50, 20)])
    # seq = aug.Sequential([aug.Add(30, 20)])
    # seq = aug.Sequential([aug.Pepper(400), aug.Salt(400)])
    # seq = aug.Sequential([aug.RandomRotate([10, 15])])
    # seq = aug.Sequential([aug.EnhanceColor(2, 'color')]) #color, contrast
    # seq = aug.Sequential([aug.RandomShear(0.1, 0.1)])
    # seq = aug.Sequential([aug.FracTranslate(), aug.Resize([256, 256]), aug.CenterCrop(224)])

    sub1 = transforms.Compose([sometimes(aug.FracTranslate()),
                              sometimes(aug.HorizontalFlip()),
                              sometimes(aug.PiecewiseAffineTransform(1, 1, 2)),
                              sometimes_low(aug.RandomRotate([10, 15])),
                              sometimes(aug.Add(-50, 20)),
                              sometimes(aug.Pepper(400)),
                              sometimes(aug.EnhanceColor(2, 'color')),
                              aug.Resize([256, 256]),
                              aug.CenterCrop(224)
                             ])
    sub2 = transforms.Compose([sometimes(aug.FracTranslate()),
                              sometimes(aug.HorizontalFlip()),
                              sometimes(aug.GaussianBlur(2)),
                              sometimes_low(aug.RandomShear(0.1, 0.1)),
                              sometimes(aug.Add(30, 20)),
                              sometimes(aug.Salt(400)),
                              sometimes(aug.EnhanceColor(2, 'contrast')),
                              aug.Resize([256, 256]),
                              aug.CenterCrop(224)
                             ])


    sub = [sub1,sub2]

    tansf = transforms.RandomChoice(sub)
    return tansf





def LoadVideoAttention(root_train, arg_train_list, root_eval, arg_test_list):
    train_dataset = read_data.FrameAttentionDataSet(
        video_root=root_train,
        video_list=arg_train_list,
        transformVideoAug=random_transformVideoAug(),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))]),
        sample_rate=3.5
    )
    val_dataset = read_data.FrameAttentionDataSet(
        video_root=root_eval,
        video_list=arg_test_list,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))]),
        sample_rate=3.5)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)

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
            plt.imshow(va[0, j, 0, :, :])
            plt.pause(1)
        pass