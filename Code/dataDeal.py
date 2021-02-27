import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil


def dataPick(data_dir):
    video_labels = []
    bin_20 = {}
    bin_40 = {}
    bin_60 = {}
    bin_80 = {}
    bin_100 = {}
    for i in data_dir:
        with open(i, "r") as f:
            imf = f.readlines()
            for id, line in enumerate(imf):
                video_label = line.strip().split(',')

                video_name, emotion, energy, fatigue, attention, motivate, Global_Status = video_label

                emotion = (np.float32(emotion) - 1) / 9.0
                energy = (np.float32(energy) - 1) / 99.0
                fatigue = (np.float32(fatigue) - 1) / 4.0
                attention = (np.float32(attention) - 1) / 4.0
                motivate = (np.float32(motivate) - 1) / 4.0
                Global_Status = (np.float32(Global_Status) - 1) / 4.0

                # video_path = os.path.join(video_root, video_name[:video_name.rfind(".")])
                a = fatigue * 100
                video_labels.append(a)
                if a < 20:
                    bin_20[video_name] = a
                elif a < 40:
                    bin_40[video_name] = a
                elif a < 60:
                    bin_60[video_name] = a
                elif a < 80:
                    bin_80[video_name] = a
                elif a <= 100:
                    bin_100[video_name] = a
    # drawHist("Data",video_labels)
    return bin_20, bin_40, bin_60, bin_80, bin_100

def randomdict(dic , a=None, b=None , c=None, d=None, e=11, num=100):
    totalvideo = []
    listn = []

    if a is None:
        listn.append('a')
        a = 0
    if b is None:
        listn.append('b')
        b = 0
    if c is None:
        listn.append('c')
        c = 0
    if d is None:
        listn.append('d')
        d = 0
    if e is None:
        listn.append('e')
        e = 0

    remainNum = num - a - b - c - d - e
    if remainNum != 0:
        randomNum = int(remainNum / len(listn))
        for i in listn:
            if i == "a":
                a = randomNum
            elif i == "b":
                b = randomNum
            elif i == "c":
                c = randomNum
            elif i == "d":
                d = randomNum
            elif i == "e":
                e = randomNum
    else:
        randomNum = 0
    for n, l in enumerate(dic):
        if n == 0:
            rn = a
        elif n == 1:
            rn = b
        elif n == 2:
            if remainNum - len(listn) * randomNum != 0:
                rn = c + (remainNum - len(listn) * randomNum)
            else :
                rn = c
        elif n == 3:
            rn = d
        elif n == 4:
            rn = e
        bins = random.sample(l.keys(), rn)
        totalvideo.extend(bins)

    return totalvideo



def drawHist(title, data):
    n, a, b = plt.hist(data, rwidth=0.5)
    print("num", n)
    print("a", a)
    print("b",  b)
    plt.title(title)
    plt.show()

def videoPick(lis, folder, aimfolder):
    for i in lis:
        if i[i.rfind("."):] != ".mp4":
            videoName = i+".mp4"
        else:
            videoName = i

        for dir in folder:
            srcpath = os.path.join(dir, videoName)
            if not os.path.exists(srcpath):
                print("%s not exit!"%srcpath)
            else:
                if not os.path.exists(aimfolder):
                    os.makedirs(aimfolder)
                dstfile = os.path.join(aimfolder, videoName)
                shutil.copyfile(srcpath,dstfile)

                lis.remove(i)
                break
    print(lis)





# video_list = r"C:\Users\ASUS\Desktop\新建文件夹\Data-Train.txt",r"C:\Users\ASUS\Desktop\新建文件夹\Data-Eval.txt"
# # video_list = r"C:\Users\ASUS\Desktop\新建文件夹\a.txt",r"C:\Users\ASUS\Desktop\新建文件夹\b.txt"
# # # video_list = r"C:\Users\ASUS\Desktop\新建文件夹\Data2-Train.txt"
# # # video_list = r"C:\Users\ASUS\Desktop\新建文件夹\Data1-Train.txt"
# # data = readScore(video_list)
# # data = dataPick(video_list)
# # tol = randomdict(data)
# tol = ['DXD3.mp4', 'Morning-2', '828888-0', 'Morning-5', '818988', 'LZ8.mp4', '918999', '1100', 'V-4.mp4', '828888-1', 'coco-3.mp4', 'GZ1.mp4', 'coco-8.mp4', 'good', 'Sleepy-JJ2', '928898', 'FC8.mp4', 'FC10.mp4', '865777', '735566', 'LZ4.mp4', 'FC18.mp4', 'Tie-3', 'Lee-1.mp4', 'Offwork-1', 'FC13.mp4', 'JiePeng-1.mp4', '857787', '466666', '574444', 'Lee-6.mp4', 'Offwork-2', 'Lee-7.mp4', 'Tie-4', 'Morning-6', 'Gohome-1', 'YouJunchu-1.mp4', 'gang9216-2.mp4', '435665', '685345', '675555', 'Tie-5', 'V-9.mp4', 'V-13.mp4', 'Morning-0', 'Sleepy32', 'coco-2.mp4', 'Tie-2', '594655', 'FC5.mp4']
# a = ['Morning-2', 'Morning-5', 'LZ8.mp4', '1100', '828888-1', 'GZ1.mp4', 'good', '928898', 'FC10.mp4', '735566', 'FC18.mp4', 'Lee-1.mp4', 'FC13.mp4', '857787', '574444', 'Offwork-2', 'Tie-4', 'Gohome-1', 'gang9216-2.mp4', '685345', 'Tie-5', 'V-13.mp4', 'Sleepy32', 'Tie-2', 'FC5.mp4']
# b = ['Morning-5', '1100', 'GZ1.mp4', '928898', '735566', 'Lee-1.mp4', '857787', 'Offwork-2', 'Gohome-1', '685345', 'V-13.mp4', 'Tie-2']
# c = ['1100', '928898', 'Lee-1.mp4', 'Offwork-2', '685345', 'Tie-2']
# d = ['928898', 'Offwork-2', 'Tie-2']
# e = ['Offwork-2']
# fol = r"F:\R_data\Data-1_Face",r"F:\R_data\Data-2_Face",r"C:\code\Data1\Viedo database-BIAI"
# videoPick(e,folder=fol,aimfolder=r"F:\R_data\smallsample1")
# pass
# # drawHist("Data",data)

def labelPick(list, txt, aim_evaltxt, aim_traintxt):
    with open(txt, "r") as f:
        imfs = f.readlines()
    with open(aim_evaltxt, "a+") as ff:
        for imf in imfs:
            if imf.strip().split(',')[0] in list:
                imfs.remove(imf)
                ff.write(imf)
                ff.flush()

    with open(aim_traintxt, "a+") as ft:

        for imf in imfs:
            ft.writelines(imf)
            ft.flush()

a = r"C:\Users\ASUS\Desktop\aaa.txt"
b = r"C:\Users\ASUS\Desktop\bbb.txt"
video_list = [r"C:\Users\ASUS\Desktop\label_avg.txt"]  # 所有样本标签
data = dataPick(video_list)
tol = randomdict(data ,a=None,b=None,c=None,d=None,e=8,num=58)
labelPick(tol,video_list[0],a,b)
print(tol)