# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']
house_labelmap = [
    0, #787878_灰色_壁
    3, #503232_茶色_床
    5, #787850_汚い緑_天井
    8, #e6e6e6_白よりの灰色_窓
    14, #08ff33_明るい緑_ドア
    82 #ffad00_オレンジ_天井のライト
]

def visualize_result(data, pred, args):
    (img, info) = data

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color),
                            axis=1).astype(np.uint8)
    img_name = info.split('/')[-1]

    save_dir = 'test_output/'
    save_img = os.path.join(args.result, img_name)
    cv2.imwrite(save_dir+save_img, pred_color)


def test(segmentation_module, loader, args):
    segmentation_module.eval()
    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(args.imgSize)

            numpy_scores = scores.cpu().numpy()
            cand_indice = house_labelmap
            selected_c = []
            for y in range(numpy_scores.shape[2]):   
                y_arr = []
                for x in range(numpy_scores.shape[3]):
                    vec = numpy_scores[0, :, y, x]
                    rank = np.argsort(vec)
                    within_top = rank[-10:]
                    if np.any(np.isin(cand_indice, within_top)):
                        selected_ind = cand_indice[np.argmax(vec[cand_indice])]
                    else:
                        selected_ind = np.argmax(vec)
                    y_arr.append(selected_ind)
                selected_c.append(y_arr)
            pred = np.array(selected_c)

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred, args)

        pbar.update(1)


def main(args):
    torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    args.test_imgs = [
#         'test_input/panel_00_ori.jpg',
#         'test_input/panel_01_ori.jpg',
#         'test_input/panel_02_ori.jpg',
#         'test_input/panel_03_ori.jpg',
#         'test_input/panel_04_ori.jpg',
#         'test_input/panel_05_ori.jpg',
#         'test_input/panel_06_ori.jpg',
#         'test_input/panel_07_ori.jpg',
#         'test_input/panel_08_ori.jpg',
#         'test_input/panel_09_ori.jpg',
#         'test_input/panel_10_ori.jpg',
#         'test_input/panel_11_ori.jpg',
#         'test_input/panel_12_ori.jpg',
#         'test_input/panel_13_ori.jpg',
#         'test_input/173_768x1536.jpg',
#         'test_input/panel_81010.jpg',
#         'test_input/panel_81013.jpg',
#         'test_input/panel_81014.jpg',
#         'test_input/panel_81021.jpg',
#         'test_input/panel_81022.jpg',
#         'test_input/panel_81028.jpg',
#         'test_input/panel_81030.jpg',
        'test_input/panel_81045.jpg',
        'test_input/panel_81049.jpg',
        'test_input/panel_81050.jpg',
        'test_input/panel_81056.jpg',
        'test_input/panel_81087.jpg',
        'test_input/panel_81109.jpg',
        'test_input/panel_81121.jpg',
        'test_input/panel_81122.jpg',
        'test_input/panel_81127.jpg',
        'test_input/panel_81138.jpg',
        'test_input/panel_81142.jpg',
        'test_input/panel_81162.jpg',
        'test_input/panel_81195.jpg',
        'test_input/panel_81410.jpg',
        'test_input/panel_84608.jpg'
    ]
    list_test = [{'fpath_img': x} for x in args.test_imgs]

    print(list_test)

    dataset_test = TestDataset(
        list_test, args, max_sample=args.num_val)
    loader_test = torchdata.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, args)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_imgs', required=True, nargs='+', type=str,
                        help='a list of image paths that needs to be tested')
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
