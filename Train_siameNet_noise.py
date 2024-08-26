import torch
import torch.nn as nn
from torch.nn import init

import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import math
from time import time
import cv2
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='SiamNet')
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

oju8parser = ArgumentParser(description='SiamNet')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=25, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')


args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912
batch_size = 64


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()


def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']
Training_labels = Training_labels
Qinit = np.linalg.pinv(Phi_input)
Iden = np.eye(n_output)
ATA_numpy = np.dot(Phi_input.transpose(), Qinit.transpose())
QinitN = np.linalg.pinv(Iden - ATA_numpy)
Noise_name = './%s/noise_%d.mat' % (args.data_dir, cs_ratio)
N = sio.loadmat('%s' % (Noise_name))
Noise = N['Noise']



def rgb2ycbcr(rgb):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    return ycbcr.reshape(shape)


# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img2)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)

    return mse

# Define SiamNet Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias1_f = nn.Parameter(torch.full([32], 0.01))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias2_f = nn.Parameter(torch.full([32], 0.01))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias1_b = nn.Parameter(torch.full([32], 0.01))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias2_b = nn.Parameter(torch.full([32], 0.01))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1_forward, padding=1, bias=self.bias1_f)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_forward, padding=1,  bias=self.bias2_f)
        x = F.relu(x)
        x = F.conv2d(x, self.conv1_backward, padding=1,  bias=self.bias1_b)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1,  bias=self.bias2_b)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        x_pred = x_pred.view(-1, 1089)



        return x_pred


# Define SiamNet
class SiamNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SiamNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))


        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb)

        x_final = x

        return x_final


model = SiamNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float(), torch.Tensor(Noise[:, index]).float()

    def __len__(self):
        return self.len


if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_SiamNet_layer_%d_group_%d_ratio_%d_lr_%.4f_unsuper_R2R0.1_Dual_noise" % (
    args.model_dir, layer_num, group_num, cs_ratio, learning_rate)

log_file_name = "./%s/Log_SiamNet_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (
    args.log_dir, layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    ckp = torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckp.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)
Iden = torch.from_numpy(Iden).type(torch.FloatTensor)
Iden = Iden.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)
QinitN = torch.from_numpy(QinitN).type(torch.FloatTensor)
QinitN = QinitN.to(device)


def together(inputs, S, H, L):
    inputs = torch.reshape(inputs, [-1, 33, 33])
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
    inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
    return inputs


mse = nn.MSELoss()
ATA = torch.mm(torch.transpose(Phi, 0, 1), torch.transpose(Qinit, 0, 1))
PhiN = Iden - ATA
# Training loop
for epoch_i in range(start_epoch, end_epoch + 1):
    for data,n in rand_loader:
        batch_x = data
        batch_x = batch_x.to(0)
        n = n.to(0)
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1)) + n
        noise = (torch.FloatTensor(Phix.size()).normal_(mean=0, std=10 / 255).cuda())
        x_output = model(Phix + 0.5*noise, Phi, Qinit)
        noise1 = (torch.FloatTensor(x_output.size()).normal_(mean=0, std=5 / 255).cuda())
        noise2 = (torch.FloatTensor(x_output.size()).normal_(mean=0, std=5 / 255).cuda())

        PhiNx = torch.mm(x_output+noise1, torch.transpose(PhiN, 0, 1))
        x_input1 = PhiNx
        x_output1 = model(x_input1, PhiN, Iden)
        loss_range = torch.mean(
            torch.pow(torch.mm(x_output1, torch.transpose(PhiN, 0, 1)) - torch.mm(x_output+noise2, torch.transpose(PhiN, 0, 1)), 2)                                                                                      )

        loss_discrepancy = torch.mean(
            torch.pow(torch.mm(x_output, torch.transpose(Phi, 0, 1)) - (Phix-noise/0.5), 2))
        loss_same = torch.mean(
            torch.pow(x_output1 - x_output, 2))

        loss_all = loss_discrepancy + 0.1 * (loss_range) + 0.05 * loss_same

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


os.makedirs("./%s/" % (model_dir), exist_ok=True)
torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters




