
#编码层数
lPosition = 10#位置编码层数
lDirection = 4#方向编码层数

#学习率参数
learn_rate = 5e-4
lr_decay = 500#学习率指数下降参数

#运行参数
totalSteps = 100000#1个epoch是100steps,共1000个epoch
render_one_test_image_epoch = 50#每50个epoch渲染一张图片
half_res = True#读取图片变为原来一半

#光束参数
chunk = 1024 * 16#同时处理的光线数量
networkChunk = 1024 * 32#渲染时同时处理的光线数量

#采样参数
N_rand = 1024#从一张训练图像中随机采样的光线数量或者说是像素数量
Nc = 64#Coarse Net一条光束的采样点数量
Nf = 128#Fine Net一条光束的采样点数量

preCrop_iter = 500#采用区域采样的steps
preCrop_fraction = 0.5#区域采样的参数

#可选参数
white_background = True#是否将背景置为白色
perturb = True#是否在范围内采取均匀采样
use_viewDirection = True#是否加入方向信息
use_FineModel = True#是否使用Fine Net

#噪音参数
raw_noise_std = 0.

#保存路径
testImg_save_pth = './fine_Image/'
model_save_pth = './models/'
renderingImg_save_path = './rendered_Image/'