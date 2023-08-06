import os
import torch
import numpy as np
import json
from PIL import Image
import torch.nn.functional as F
import imageio
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter
from MyModel import *
from MyConfig import *

###############################
#       图片数据和相机参数加载     #
################################
#Z轴平移矩阵
translate_positive_z = lambda z: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, z],
    [0, 0, 0, 1]
], dtype=torch.float32)
#世界坐标系x轴逆时针旋转
rotate_WCS_x_CCW = lambda phi: torch.tensor([
    [1, 0,  0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

rotate_WCS_y_CCW = lambda theta: torch.tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0,  np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
#x轴反向并交换yz轴
change_WCS_yz = torch.tensor([
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):#获取球面位姿
    c2w = translate_positive_z(radius)
    c2w = rotate_WCS_x_CCW(phi / 180. * np.pi) @ c2w
    c2w = rotate_WCS_y_CCW(theta / 180. * np.pi) @ c2w
    c2w = change_WCS_yz @ c2w
    return c2w

#加载数据
def load_blender_data(dirpath, half_res=False, testSkip=1, renderSize=40, renderAngle=30.0):
    #输出Img, Pose, RenderPose, [H, W, focal], index_split(即图片、位姿、渲染位姿、[高, 宽, 焦距]
    #Img shape = [400,H, W, 4],400是照片数量,H和W是照片高宽,4是RGBA通道
    #Pose  shape =[400, 4, 4]
    #renderAngle要预测渲染的角度
    splits = ['train', 'val', 'test']#照片划分为训练集、验证集和测试集
    jsons = {}
    #将三个集合的json读入jsons中
    for s in splits:
        with open(os.path.join(dirpath, 'transforms_{}.json'.format(s)), 'r') as f:
            jsons[s] = json.load(f)

    allImg = []
    allPose = []
    counts = [0]
    for s in splits:
        if s == 'train' or testSkip == 0:
            skip = 1
        else:
            skip = testSkip

        jsonData = jsons[s]
        Imgs = []
        Poses = []
        for frame in jsonData['frames'][::skip]:#[::skip]表示每skip个选一个
            #print(frame)
            #break
            file_path = frame['file_path'].replace('./', '')#去掉路径中的'./'
            matrix = np.array(frame['transform_matrix'], dtype=np.float32)#相机参数矩阵,位姿
            img = Image.open(os.path.join(dirpath, file_path + '.png'))#获取图片数据
            if half_res:
                H, W = img.height, img.width
                H = H // 2
                W = W // 2
                img = img.resize((H, W), resample=Image.LANCZOS)

            img = np.array(img, dtype=np.float32) / 255.#照片数据归一化到[0,1]
            Imgs.append(img)
            Poses.append(matrix)

        counts.append(counts[-1]+len(Imgs))#记录每一集合中第一个索引
        allImg.append(Imgs)
        allPose.append(Poses)

    #print(np.array(allImg[2]).shape)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]#将索引分成3类
    allImg = np.concatenate(allImg, axis=0)#在类别维度上合并即100的train,100的val和200的test合并为400
    #print(np.array(allImg).shape)
    allPose = np.concatenate(allPose, axis=0)

    H, W = allImg[0].shape[:2]
    camera_angle_x = jsons['train']['camera_angle_x']#获取相机angle
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)#计算相机焦距

    render_poses = torch.stack([pose_spherical(theta, -renderAngle, 4.0)
                                for theta in np.linspace(-180, 180, renderSize + 1)[:-1]], dim=0)#linespace在(-180,180)[:-1]上产生reandersize个点,pose_spherical返回一个4x4矩阵

    return allImg, allPose, render_poses, [H, W, focal], i_split

#Utils
#位置编码的类
class PositionalEncoding:
    def __init__(self, multiRes, includeInput=True, dim=3):
        self.embed_fns = []
        self.totalDims = 0
        encode_fn = [torch.sin, torch.cos]
        if includeInput:
            self.embed_fns.append(lambda x: x)
            self.totalDims += dim
        for res in range(multiRes):
            res = 2 ** res
            for fn in encode_fn:
                self.embed_fns.append(lambda x, fn_=fn, res_=res: fn_(res_ * x))#公式
                self.totalDims += dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)#将所有数对拼接成一行

#获取光线,包括原点和方向
def get_rays(H, W, K, c2w, device):
    (x, y) = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')

    #(x, y) = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
    dirs = torch.stack([(x - K[0][2]) / K[0][0], -(y - K[1][2]) / K[1][1], -torch.ones_like(x)], dim=-1)  # (H,W,3)
    dirs = dirs.to(device)
    rays_d = dirs @ (c2w[:3, :3].t())
    # Same as: rays_d=torch.sum(dirs[...,None,:]*c2w[:3,:3],dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

#获取每束光线的coarse采样点
def get_tVals(batch_size, sample_size, near=2., far=6., lindisp=True, perturb=True):
    # batch_size光线数量,sample_size样本点数, near和far表示一个相机的取景范围
    near = torch.tensor(near, dtype=torch.float32).expand((batch_size, sample_size))
    far = torch.tensor(far, dtype=torch.float32).expand((batch_size, sample_size))

    tVals = torch.linspace(0., 1., steps=sample_size)#生成(0., 1.)之间的sample_size个点
    if lindisp:
        tVals = 1. / (1. / near * (1 - tVals) + 1. / far * tVals)
    else:
        tVals = near + (far - near) * tVals#将0.~1.的点转换到near-far, shape=[batch_size, sample_size]
    # tVals shape: (Batch,Nc)

    if perturb:
        mid = (tVals[..., 1:] + tVals[..., :-1]) * 0.5#获取相邻样本点之间的中间点, shape=[batch_size, sample_size-1]
        above = torch.cat([mid, tVals[..., -1:]], dim=-1)#mid和最后一列即最大值拼接,保持形状为[batch_size, sample_size]
        below = torch.cat([tVals[..., :1], mid], dim=-1)#mid和最小值拼接
        tRand = torch.rand((batch_size, sample_size))#0-1之间的随机数,shape=[batch_size, sample_size]
        tVals = below + tRand * (above - below)#得到sample_size个随机点,shape = [batch_size, sample_size]
    return tVals

#渲染的值即颜色、深度等
def VolumeRender(raw_input, t_vals, rays_d, device, raw_noise_std=0., white_backGround=False):
    # raw_input shape = [ray_size, sample_size, 4]
    # t_vals shape = [ray_size, sample_size]
    # rays_d shape=[ray_size, 3]

    raw_RGB = raw_input[..., :3]  # (ray_size,sample_size,3)
    raw_sigma = raw_input[..., 3]  # (ray_size,sample_size)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_sigma.shape) * raw_noise_std
    sigma = F.relu(raw_sigma + noise)
    RGB = torch.sigmoid(raw_RGB)

    delta = t_vals[..., 1:] - t_vals[..., :-1]  # 获取相邻采样点的间隔
    delta = torch.cat([delta, torch.tensor(1e10, dtype=torch.float32, device=device).expand(delta[..., :1].shape)],
                      dim=-1)  # (ray_size, sample_size)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    exponentialTerm = torch.exp(-sigma * delta)
    alpha = 1 - exponentialTerm

    Transmittance = torch.cat(
        [torch.ones_like(exponentialTerm[..., :1]), torch.cumprod(exponentialTerm + 1e-10, dim=-1)],
        dim=-1)[..., :-1]  # 前面采样点的阻碍度即采样点的光线透过度,shape=[rays_size, samples_size]
    weight = Transmittance * alpha  # 采样点对像素颜色的贡献度

    RGB_map = torch.sum(weight[..., None] * RGB, dim=-2)  # shape=[rays_size, 3],论文颜色公式
    depth_map = torch.sum(weight * t_vals, dim=-1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weight, -1))
    acc_map = torch.sum(weight, -1)

    if white_backGround:
        RGB_map = RGB_map + (1. - acc_map[..., None])

    return RGB_map, disp_map, acc_map, weight, depth_map

#获取fine采样点
def sample_pdf(bins, weight, N_sample, device, perturb=True):

    # bins shape=[rays_size, Nc -1], weight shape=[ray_size, Nc-2], N_sample = Nf

    weight = weight + 1e-5  # 保证weight非nan
    pdf = weight / torch.sum(weight, dim=-1, keepdim=True)  # 将weight归一化到[0,1]
    cdf = torch.cumsum(pdf, dim=-1)  # 将weight由pdf概率分布转换为cdf概率分布
    cdf = torch.cat([torch.zeros_like(weight[..., :1]), cdf], dim=-1)  # 用0拼接使cdf分布的shape变为[rays_size, Nc-1]
    NcMinusOne = cdf.shape[-1]  # shape=[Nc-1]

    if perturb:
        # 从0-1的均匀分布中随机挑选数据
        u = torch.rand(list(weight.shape[:-1]) + [N_sample])
        # shape=[rays_size, N_sample],list(weight.shape[:-1])的结果是shape=[rays_size]
    else:
        u = torch.linspace(0., 1., steps=N_sample)
        u = u.expand(list(weight.shape[:-1]) + [N_sample])

    u = u.contiguous()  # (Batch, Nf)
    u = u.to(device)
    idxs = torch.searchsorted(cdf, u, right=True)  # 在递增序列cdf中返回一个cdf中值大于等于u中对应行的元素值的索引,形状和u形状一致, shape=[rays_size, Nf]
    below = torch.max(torch.zeros_like(idxs), idxs - 1)
    above = torch.min(torch.ones_like(idxs) * (NcMinusOne - 1), idxs)
    inds_g = torch.stack([below, above], dim=-1)  # shape = [rays_size, Nf, 2]

    matched_shape = list(inds_g.shape[:-1]) + [NcMinusOne]  # shape=[rays_size, Nf, Nc-1]
    cdf_g = torch.gather(cdf[..., None, :].expand(matched_shape), dim=-1, index=inds_g)
    # 根据索引挑选数据,逆采样,相当于从概率大的位置多做采样点
    # torch.gather中cdf为值矩阵,index为索引矩阵,若index为行向量,则(行号,index值)构成索引,dim=0则交换索引内双方位置,dim=1不动,然后按照索引从value中取值
    bins_g = torch.gather(bins[..., None, :].expand(matched_shape), dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]  # hape=[ray_size, Nf],相当于采样盒子的大小,cdf_g[..., 1]中放的是above是大值
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples



#Train
#运行fn
def batchify(fn, netChunk=None):
    if netChunk is None:
        return fn
    else:
        def fn_(x):
            return torch.cat([fn(x[i:i + netChunk]) for i in range(0, x.shape[0], netChunk)], dim=0)

        return fn_



def render(rays, Coarse, Fine, posENC, dirENC, perturb, device):
    # rays shape is [size, 6 or 9]
    ray_size = rays.shape[0]
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]  # (ray_size,3), (ray_size,3)
    viewDir = rays[..., 6:] if use_viewDirection else None  # (ray_size,3)
    # 获得每个光束上的粗采样点
    tVals = get_tVals(batch_size=ray_size, sample_size=Nc, near=2., far=6., lindisp=False,
                      perturb=perturb)  # (ray_size,Nc)
    tVals = tVals.to(device)
    # 论文公式
    points = rays_o[..., None, :] + rays_d[..., None, :] * tVals[..., None]  # (ray_size,Nc,3)

    # Run Coarse Network
    points_coarse_shape = points.shape
    points = torch.reshape(points, [-1, 3])
    embedded = posENC(points)  # (ray_size*Nc, 2*lPosition*3+3)#对输入的点位置进行编码
    if use_viewDirection:#加入方向信息
        viewDir_ = viewDir[..., None, :].expand(points_coarse_shape)
        viewDir_ = torch.reshape(viewDir_, [-1, 3])
        embedded = torch.cat([embedded, dirENC(viewDir_)], dim=-1)  # (ray_size*Nc, 2*lPosition*3+3 + 2*lDirection*3+3)

    outputs = batchify(Coarse, networkChunk)(embedded)  #返回shape=[ray_size *Nc, 4] ,前三个为RGB,最后为sigma
    outputs = torch.reshape(outputs, shape=list(points_coarse_shape[:-1]) + [outputs.shape[-1]]) #shape=[batch_size, Nc, 4]最后的4表示将采样点和该店的体密度拼接

    RGB_coarse, disp_coarse, acc_coarse, weights, depth_coarse = \
        VolumeRender(outputs, tVals, rays_d, device, raw_noise_std, white_background)

    # Run Fine Network
    if use_FineModel:
        tValsMid = (tVals[..., 1:] + tVals[..., :-1]) * 0.5#相邻粗采样点的间隔
        tValsFine = sample_pdf(tValsMid, weights[..., 1:-1], Nf, device, perturb)
        tValsFine = tValsFine.detach()

        tValsFine, _ = torch.sort(torch.cat([tVals, tValsFine], dim=-1), dim=-1)  # shape = [rays_size, Nc+Nf]

        points = rays_o[..., None, :] + rays_d[..., None, :] * tValsFine[..., None]  # (ray_size,Nc+Nf,3)

        points_fine_shape = points.shape
        points = torch.reshape(points, [-1, 3])
        embedded = posENC(points)  # (ray_size*(Nc+Nf), 2*lPosition*3+3)
        if use_viewDirection:
            viewDir_ = viewDir[..., None, :].expand(points_fine_shape)
            viewDir_ = torch.reshape(viewDir_, [-1, 3])
            embedded = torch.cat([embedded, dirENC(viewDir_)], dim=-1)
        outputs = batchify(Fine, networkChunk)(embedded)
        outputs = torch.reshape(outputs, shape=list(points_fine_shape[:-1]) + [outputs.shape[-1]])

        RGB_fine, disp_fine, acc_fine, weights, depth_fine = \
            VolumeRender(outputs, tValsFine, rays_d, device, raw_noise_std, white_background)
        ret = {'rgb_map': RGB_fine, 'disp_map': disp_fine, 'acc_map': acc_fine, 'depth_map': depth_fine,
               'rgb_coarse': RGB_coarse, 'disp_coarse': disp_coarse, 'acc_coarse': acc_coarse,
               'depth_coarse': depth_coarse}
    else:
        ret = {'rgb_map': RGB_coarse, 'disp_map': disp_coarse, 'acc_map': acc_coarse, 'depth_map': depth_coarse}

    return ret

def render_full_image(render_pose, hw, K, Coarse, Fine, posENC, dirENC, device):
    H, W = hw
    rays_o, rays_d = get_rays(H, W, K, render_pose, device)  # (H,W,3)
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])
    rays = torch.cat([rays_o, rays_d], dim=-1)
    if use_viewDirection:
        viewDir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays = torch.cat([rays, viewDir], dim=-1)

    all_ret = {}
    #渲染图片,一次处理chunk条
    for i in tqdm(range(0, rays.shape[0], chunk), desc='Rendering Image', leave=False):
        ret = render(rays[i:i + chunk], Coarse, Fine, posENC, dirENC, perturb=False, device=device)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret




def Train(dataSetPath, exp_name, test_img_idx):
    print('--Load Data and Preprocess Data--')
    image, poses, renderPoses, hwf, idx_split = load_blender_data(dataSetPath, half_res=False, renderSize=40,
                                                                  renderAngle=30.0)
    # load_blender_data(dataSetPath, half_res=False, renderSize=40, renderAngle=30.0)其中renderPoses用于旋转展示的位姿
    # image shape [400, H, W, 4], pose shape [400, 4, 4], renderPoses shape [renderSize, 4, 4]

    idx_train, idx_val, idx_test = idx_split
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    if white_background:#是否将照片背景设置为白色
        # 将白色背景绘制到图片上并生成3通道图片
        image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
    else:
        image = image[..., :3]

    poses = torch.tensor(poses).to(device)
    print('--Data Preprocession Finished--')
    print('--Start Position Encode--')
    posENC = PositionalEncoding(lPosition)
    dirENC = PositionalEncoding(lDirection) if use_viewDirection else None
    direction_ch = dirENC.totalDims if use_viewDirection else 0
    #加载Coarse Net
    Coarse = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                  direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(device)
    grad_vars = list(Coarse.parameters())

    #使用Fine Net
    if use_FineModel:
        Fine = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                    direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(device)
        grad_vars += list(Fine.parameters())
    else:
        Fine = False

    #优化器，损失函数以及学习下降速率参数
    optimizer = torch.optim.Adam(params=grad_vars, lr=learn_rate, betas=(0.9, 0.999))
    decay_rate = 0.1
    decay_steps = lr_decay * 1000
    mseLoss = torch.nn.MSELoss()
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
    testPSNR = PeakSignalNoiseRatio(data_range=1.0)
    print('--Position Encode Finished--')

    print('--Train Start--')

    epochTQDM = tqdm(range(1, totalSteps + 1))
    writer = SummaryWriter('runs/' + exp_name)
    totalLoss = 0.
    totalPSNR = 0.
    bestPSNR = 0.

    for step in epochTQDM:
        random_idx = np.random.choice(idx_train)
        target = image[random_idx]
        target = torch.tensor(target).to(device)
        pose = poses[random_idx]

        rays_o, rays_d = get_rays(H, W, K, pose, device)  # (H,W,3), (H,W,3)

        if step < preCrop_iter:#在区域内采样
            dH = int(0.5 * H * preCrop_fraction)
            dW = int(0.5 * W * preCrop_fraction)
            coords = torch.stack(torch.meshgrid(torch.arange(H // 2 - dH, H // 2 + dH),
                                                torch.arange(W // 2 - dW, W // 2 + dW), indexing='ij'), dim=-1)
        else:#整个照片范围内采样
            coords = torch.stack(torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij'), dim=-1)
        coords = torch.reshape(coords, shape=[-1, 2])#保证列数为2，行数不定。2维表示行列位置
        batch_ray_idxs = np.random.choice(coords.shape[0], size=N_rand, replace=False)#选中像素数量也就是光束的数量
        selected_coords = coords[batch_ray_idxs].long()  # (N_rand,2)
        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)找到对应像素位置的光源
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)找到对应像素的光束方向
        target = target[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)目标图像中对应位置的像素颜色

        rays = torch.cat([rays_o, rays_d], dim=-1)#[N_rand, 6]
        if use_viewDirection:
            viewDir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.cat([rays, viewDir], dim=-1)#方向信息进行拼接

        render_return = render(rays, Coarse, Fine, posENC, dirENC, perturb=perturb, device=device)
        pred = render_return['rgb_map']

        optimizer.zero_grad()
        loss = mseLoss(pred, target)

        if use_FineModel:
            loss_coarse = mseLoss(render_return['rgb_coarse'], target)
            loss += loss_coarse
        psnr = PSNR(pred, target)
        loss.backward()
        optimizer.step()
        totalPSNR += psnr.item()
        totalLoss += loss.item()

        new_lr = learn_rate * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        #print(new_lr)

        #渲染测试图片
        if step % (render_one_test_image_epoch * 100) == 0:
            with torch.no_grad():
                Coarse.eval()
                Fine.eval()
                render_return = render_full_image(poses[idx_test[test_img_idx]], [H, W], K, Coarse, Fine, posENC,
                                                  dirENC, device)
                pred_image = torch.reshape(render_return['rgb_map'].cpu(), [H, W, 3])
                target_image = torch.tensor(image[idx_test[test_img_idx]])
                psnr = testPSNR(pred_image, target_image)
                pred_image = (255. * pred_image).to(torch.uint8).numpy()
                imageio.imsave(testImg_save_pth + '/{:05d}_{:.2f}.png'.format(int(step // 100), psnr), pred_image)
            writer.add_scalar('PSNR_test', psnr, int(step // 100))

            #保存训练中最好的模型参数
            if bestPSNR < psnr:
                bestPSNR = psnr
                save_path = model_save_pth + '/Epoch_{}.tar'.format(int(step // 100))
                torch.save({
                    'step': step,
                    'Coarse': Coarse.state_dict(),
                    'Fine': Fine.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, save_path)
            Coarse.train()
            Fine.train()

        # 每一个epoch即100step更新tqdm
        if (step % 100) == 0:
            avgPSNR = totalPSNR / 100.
            avgLoss = totalLoss / 100.
            epoch = int(step // 100)
            epochTQDM.set_postfix({
                'epoch': epoch,
                'loss': '{:.04f}'.format(avgLoss),
                'psnr': '{:.02f}'.format(avgPSNR)
            })
            loss_list.append(avgLoss)
            psnr_list.append(avgPSNR)
            totalLoss = 0.
            totalPSNR = 0.
            writer.add_scalar('Loss', avgLoss, epoch)
            writer.add_scalar('PSNR_train', avgPSNR, epoch)


if __name__ == '__main__':
    print('--NeRF Program--')
    if torch.cuda.is_available():
        print('GPU is available!')
        device = torch.device('cuda:0')
    else:
        print('CPU is available!')
        device = torch.device('cpu')

    data_path = './nerf_synthetic/lego'
    expName = 'lego_nerf'
    testImg_save_pth = testImg_save_pth + expName
    model_save_pth = model_save_pth + expName
    os.makedirs(testImg_save_pth, exist_ok=True)
    os.makedirs(model_save_pth, exist_ok=True)
    test_image_index = 55
    loss_list = []
    psnr_list = []
    Train(data_path, expName, test_image_index)

    with open("./loss.txt", 'w') as f1:
        f1.write(str(loss_list))
        f1.flush()
        f1.close()

    with open("./psnr.txt", 'w') as f2:
        f2.write(str(psnr_list))
        f2.flush()
        f2.close()