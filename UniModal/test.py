import os
from argparse import ArgumentParser
from model import *
from losses import *
import datagenerators
import scipy.io as sio
from utils import dice

def test(gpu,atlas_file, model_file):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = atlas['vol']
    atlas_vol = atlas_vol[np.newaxis, ..., np.newaxis]
    atlas_seg = atlas['seg']

    # Test file and anatomical labels we want to evaluate
    test_file = open('../data/test_HCPChild.txt')
    test_strings = test_file.readlines()
    test_strings = [x.strip() for x in test_strings]
    good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

    # Set up model
    criterion = LossFunction_mpr().cuda()
    model = MPR_net_HO(criterion)
    model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    # set up atlas tensor
    input_fixed = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed = input_fixed.permute(0, 4, 1, 2, 3)

    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf.to(device)


    for k in range(0, len(test_strings)):

        vol_name, seg_name = test_strings[k].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        input_moving = torch.from_numpy(X_vol).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)

        warp, flow, flow1, refine_flow1, flow2, refine_flow2 = model(input_moving, input_fixed)

        # Warp segment using flow
        moving_seg = torch.from_numpy(X_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 4, 1, 2, 3)
        warp_seg = trf(moving_seg, flow).detach().cpu().numpy()

        vals, labels = dice(warp_seg, atlas_seg, labels=good_labels, nargout=2)
        print(np.mean(vals))





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--atlas_file", type=str, dest="atlas_file",
                        default='../data/atlas_norm.npz')
    parser.add_argument("--model_file", type=str, dest="model_file",
                        default='../models/MPR-T1-to-T1atlas-HCP/1200.ckpt',
                        help="model weight file")
    test(**vars(parser.parse_args()))

