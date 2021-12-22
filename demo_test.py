import time
import argparse
import scipy.io
from utils.utils import *
from utils.imresize import *
from model import Net
import numpy as np
import imageio
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=2, help="upscale factor")
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--input_dir', type=str, default='./input/')
    parser.add_argument('--save_path', type=str, default='./output/')

    return parser.parse_args()

def demo_test(cfg):

    net = Net(cfg.angRes, cfg.upfactor)
    net.to(cfg.device)
    model_name = 'DistgSSR_' + str(cfg.upfactor) + 'xSR_' + str(cfg.angRes) + 'x' + str(cfg.angRes)
    model = torch.load('./log/' + model_name + '.pth.tar', map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    scene_list = os.listdir(cfg.input_dir)

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.input_dir + scenes + '/view_01_01.png')
        lf_rgb_lr = np.zeros(shape=(cfg.angRes, cfg.angRes, temp.shape[0], temp.shape[1], 3))
        lf_rgb_sr = np.zeros(shape=(cfg.angRes, cfg.angRes, cfg.upfactor * temp.shape[0], cfg.upfactor * temp.shape[1], 3)).astype('float32')

        for u in range(cfg.angRes):
            for v in range(cfg.angRes):
                temp = imageio.imread(cfg.input_dir + scenes + '/view_%.2d_%.2d.png' % (u+1, v+1))
                lf_rgb_lr[u, v, :, :, :] = temp

        lf_y_lr = (0.256789 * lf_rgb_lr[:,:,:,:,0] + 0.504129 * lf_rgb_lr[:,:,:,:,1] + 0.097906 * lf_rgb_lr[:,:,:,:,2] + 16).astype('float32')
        lf_cb_lr = (-0.148223 * lf_rgb_lr[:,:,:,:,0] - 0.290992 * lf_rgb_lr[:,:,:,:,1] + 0.439215 * lf_rgb_lr[:,:,:,:,2] + 128).astype('float32')
        lf_cr_lr = (0.439215 * lf_rgb_lr[:,:,:,:,0] - 0.367789 * lf_rgb_lr[:,:,:,:,1] - 0.071426 * lf_rgb_lr[:,:,:,:,2] + 128).astype('float32')

        lf_cb_sr = np.zeros(shape=(cfg.angRes, cfg.angRes, cfg.upfactor * temp.shape[0], cfg.upfactor * temp.shape[1])).astype('float32')
        lf_cr_sr = np.zeros(shape=(cfg.angRes, cfg.angRes, cfg.upfactor * temp.shape[0], cfg.upfactor * temp.shape[1])).astype('float32')
        for u in range(cfg.angRes):
            for v in range(cfg.angRes):
                lf_cb_sr[u, v, :, :] = imresize(lf_cb_lr[u, v, :, :], cfg.upfactor)
                lf_cr_sr[u, v, :, :] = imresize(lf_cr_lr[u, v, :, :], cfg.upfactor)

        data = rearrange(lf_y_lr, 'u v h w -> (u h) (v w)')
        data = torch.from_numpy(data) / 255.0

        if cfg.crop == False:
            lf_y_sr = net(data.squeeze().to(cfg.device))
            lf_y_sr = lf_y_sr.squeeze()

        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes, vw // cfg.angRes
            subLFin = LFdivide(data, cfg.angRes, patchsize, stride)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes * patchsize * cfg.upfactor, cfg.angRes * patchsize * cfg.upfactor)

            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        out = net(tmp.to(cfg.device))
                        subLFout[u, v, :, :] = out.squeeze()

            lf_y_sr = LFintegrate(subLFout, cfg.angRes, patchsize * cfg.upfactor, stride * cfg.upfactor, h0 * cfg.upfactor, w0 * cfg.upfactor)

        lf_y_sr = 255 * lf_y_sr.data.cpu().numpy()
        lf_rgb_sr[:, :, :, :, 0] = 1.164383 * (lf_y_sr - 16) + 1.596027 * (lf_cr_sr - 128)
        lf_rgb_sr[:, :, :, :, 1] = 1.164383 * (lf_y_sr - 16) - 0.391762 * (lf_cb_sr - 128) - 0.812969 * (lf_cr_sr - 128)
        lf_rgb_sr[:, :, :, :, 2] = 1.164383 * (lf_y_sr - 16) + 2.017230 * (lf_cb_sr - 128)

        lf_rgb_sr = np.clip(lf_rgb_sr, 0, 255)
        output_path = cfg.save_path + scenes
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        for u in range(cfg.angRes):
            for v in range(cfg.angRes):
                imageio.imwrite(output_path + '/view_%.2d_%.2d.png' % (u+1, v+1), np.uint8(lf_rgb_sr[u, v, :, :]))

        print('Finished! \n')


if __name__ == '__main__':
    cfg = parse_args()
    demo_test(cfg)