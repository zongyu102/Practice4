import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def data_read(dir_path):#读取一维数组
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

def draw(filepath, savename):
    y = data_read(filepath)
    x = range(len(y))
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel(savename)
    plt.plot(x, y, color='blue', linestyle='solid', label=savename)
    if savename == 'train_loss':
        plt.title('Train Loss')
    elif savename == 'train_psnr':
        plt.title('Train psnr')


    path = './DataImage'
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = path + '/' + savename + '.png'
    plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    L_train_loss = './loss.txt'
    L_train_loss_name = 'train_loss'
    draw(L_train_loss, L_train_loss_name)

    L_train_acc = './psnr.txt'
    L_train_acc_name = 'train_psnr'
    draw(L_train_acc, L_train_acc_name)


