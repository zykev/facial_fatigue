import matplotlib.pyplot as plt
import numpy as np
import os
from torchsummary import summary
import torch
from Code_tomse2 import Model_Parts

def drawLoss(file_name, png_path, is_save=False):
    train_loss = []
    valid_loss = []
    
    train_acc = []
    valid_acc = []
    x_axis = []
    with open(file_name, "r+") as f:
        df = f.readlines()
        for i,item in enumerate(df):
            data = item.strip().split()
            train_loss.append(float(data[0]))
            valid_loss.append(float(data[1]))
            train_acc.append(float(data[2]))
            valid_acc.append(float(data[3]))
            x_axis.append(i+1)
    
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    plt.plot(x_axis,train_loss,color="b",label="Train")
    plt.plot(x_axis,valid_loss,color="y",label="Valid")
    plt.tick_params(axis="both",labelsize=14)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x_axis,train_acc,color="b",label="Train")
    plt.plot(x_axis,valid_acc,color="y",label="Valid")
    plt.tick_params(axis="both",labelsize=14)
    plt.legend()

    if is_save == True:
        if os.path.exists(png_path+".png"):
            print("There is this png!")
            png_path = png_path + "_z"
        else:
            print(png_path)
        plt.savefig(png_path)

    plt.show()
    
def drawLoss2(file_name, png_path='', is_save=False):
    train_loss = []
    valid_loss = []
    train_mseloss = []
    valid_mseloss = []
    train_acc = []
    valid_acc = []
    x_axis = []
    with open(file_name, "r+") as f:
        df = f.readlines()
        for i,item in enumerate(df):
            data = item.strip().split()
            train_loss.append(float(data[0]))
            valid_loss.append(float(data[1]))
            train_mseloss.append(float(data[2]))
            valid_mseloss.append(float(data[3]))
            train_acc.append(float(data[4]))
            valid_acc.append(float(data[5]))
            x_axis.append(i+1)
    
    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1)
    plt.plot(x_axis,train_loss,color="b",label="Train_loss")
    plt.plot(x_axis,valid_loss,color="y",label="Valid_loss")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(x_axis,train_mseloss,color="b",label="Train_mseloss")
    plt.plot(x_axis,valid_mseloss,color="y",label="Valid_mseloss")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(x_axis,train_acc,color="b",label="Train_acc")
    plt.plot(x_axis,valid_acc,color="y",label="Valid_acc")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()

    if is_save == True:
        if os.path.exists(png_path+".png"):
            print("There is this png!")
            png_path = png_path + "_z"
        else:
            print(png_path)
        plt.savefig(png_path)

    plt.show()

def drawLoss3(file_name, png_path='', is_save=False):
    train_loss = []
    valid_loss = []
    train_celoss = []
    valid_celoss = []
    train_mseloss = []
    valid_mseloss = []
    train_acc = []
    valid_acc = []
    x_axis = []
    with open(file_name, "r+") as f:
        df = f.readlines()
        for i,item in enumerate(df):
            data = item.strip().split()
            train_loss.append(float(data[0]))
            valid_loss.append(float(data[1]))
            train_celoss.append(float(data[2]))
            valid_celoss.append(float(data[3]))
            train_mseloss.append(float(data[4]))
            valid_mseloss.append(float(data[5]))
            train_acc.append(float(data[6]))
            valid_acc.append(float(data[7]))
            x_axis.append(i+1)
    
    plt.figure(figsize=(16,16))
    plt.subplot(2,2,1)
    plt.plot(x_axis,train_loss,color="b",label="Train_loss")
    plt.plot(x_axis,valid_loss,color="y",label="Valid_loss")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.plot(x_axis,train_celoss,color="b",label="Train_celoss")
    plt.plot(x_axis,valid_celoss,color="y",label="Valid_celoss")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()
    
    plt.subplot(2,2,3)
    plt.plot(x_axis,train_mseloss,color="b",label="Train_mseloss")
    plt.plot(x_axis,valid_mseloss,color="y",label="Valid_mseloss")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()
    
    plt.subplot(2,2,4)
    plt.plot(x_axis,train_acc,color="b",label="Train_acc")
    plt.plot(x_axis,valid_acc,color="y",label="Valid_acc")
    plt.tick_params(axis="both",labelsize=10)
    plt.legend()

    if is_save == True:
        plt.savefig(png_path)

    plt.show()


def load_model(dir_model):

    _structure = Model_Parts.FullModal_VisualFeatureAttention(num_class=10, feature_dim=256, at_type='nonLocal')
    model = Model_Parts.LoadParameter(_structure, dir_model)

    return model


if __name__ == "__main__":

    file_name = r"E:/onlyfat_selfa3D_2/01-18_21-04.txt"
    drawLoss2(file_name)
    
    dir_model = '/home/biai/BIAI/mood/onlyfat_selfa3DS_class_tomse2/model/12-31_18-30/epoch5_loss_0.0702_acc_0.253'
    model = load_model(dir_model)
    model.to('cuda')
    summary(model, (3,32,224,224), device='cuda')
