import time
import argparse
import scipy.io
import torch.backends.cudnn as cudnn
from utils.utils import *
from model import Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=2, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='DistgSSR_2xSR_5x5')
    parser.add_argument('--testset_dir', type=str, default='../Data/Test_2xSR_5x5/')
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--minibatch", type=int, default=12, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--save_path', type=str, default='./Results/')

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
        data = data.to(cfg.device)
        label = label.squeeze()

        if cfg.crop == False:
            outLF = net(data.to(cfg.device))
            outLF = outLF.squeeze()

        else:
            lf_lr = rearrange(data.squeeze(), '(u h) (v w) -> u v h w', u=cfg.angRes, v=cfg.angRes)
            patchsize = cfg.patchsize
            stride = patchsize // 2
            sub_lfs = LFdivide(lf_lr, patchsize, stride)

            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c (u h) (v w)')
            mini_batch = cfg.minibatch
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    input_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :]
                    out_lfs.append(net(input_lfs.to(cfg.device)))
                if (n1 * n2) % mini_batch:
                    input_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :]
                    out_lfs.append(net(input_lfs.to(cfg.device)))

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) c (u h) (v w) -> n1 n2 u v c h w', n1=n1, n2=n2, u=cfg.angRes,
                                v=cfg.angRes)
            outLF = LFintegrate(out_lfs, patchsize * cfg.upfactor, patchsize * cfg.upfactor // 2)
            outLF = outLF[:, :, 0: lf_lr.shape[2] * cfg.upfactor, 0: lf_lr.shape[3] * cfg.upfactor]

        psnr, ssim = cal_metrics(label.to(cfg.device), outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        save_path = cfg.save_path + cfg.model_name
        if not (os.path.exists(save_path + '/' + test_name)):
            os.makedirs(save_path + '/' + test_name)
        scipy.io.savemat(save_path + '/' + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                         {'LF': outLF.cpu().numpy()})
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    cfg = parse_args()
    test_on_datasets(cfg)
