import time
import argparse
import scipy.io
import torch.backends.cudnn as cudnn
from utils.utils import *
from model import Net
from einops import rearrange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=2, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='DistgSSR_2xSR_5x5')
    parser.add_argument('--testset_dir', type=str, default='../Data/Test_2xSR_5x5/')
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--save_path', type=str, default='./Results/')
    parser.add_argument("--minibatch_test", type=int, default=16, help="size of minibatch for inference")

    return parser.parse_args()


def test_on_datasets(cfg):

    net = Net(cfg.angRes, cfg.upfactor)
    net.to(cfg.device)
    cudnn.benchmark = True
    model_path = './log/' + cfg.model_name + '.pth.tar'
    if os.path.isfile(model_path):
        model = torch.load(model_path, map_location={'cuda:1': cfg.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.load_model))

    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = valid(test_loader, test_name, net)
            print('Dataset----%15s,\t PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile = open(cfg.save_path + cfg.model_name + '/Metrics.txt', 'a')
            txtfile.write('Dataset----%15s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()
            pass
        pass


def valid(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                outLF = outLF.squeeze()
        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            data = data.squeeze()

            ''' Crop LFs into Patches '''
            subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.patchsize // 2)
            numU, numV, H, W = subLFin.shape
            subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
            subLFout = torch.zeros(numU * numV, 1, cfg.angRes * cfg.patchsize * cfg.upscale_factor,
                                   cfg.angRes * cfg.patchsize * cfg.upscale_factor)

            ''' SR the Patches '''
            mini_batch = cfg.minibatch_test
            for i in range(0, numU * numV, mini_batch):
                tmp = subLFin[i:min(i + mini_batch, numU * numV), :, :, :]
                with torch.no_grad():
                    net.eval()
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[i:min(i + mini_batch, numU * numV), :, :, :] = out
            subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)
            outLF = LFintegrate(subLFout, cfg.angRes, patchsize * cfg.upfactor, stride * cfg.upfactor)

        psnr, ssim = cal_metrics(label.to(cfg.device), outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        save_path = cfg.save_path + cfg.model_name
        if not (os.path.exists(save_path + '/' + test_name)):
            os.makedirs(save_path + '/' + test_name)
        scipy.io.savemat(save_path + '/' + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                         {'LF': outLF.numpy()})
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    cfg = parse_args()
    test_on_datasets(cfg)
