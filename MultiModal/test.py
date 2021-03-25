from argparse import ArgumentParser
import datagenerators
from models import *
from losses import *
from utils import *
import nibabel as nib
from datagenerators import load_volfile

def test(gpu, init_model_file, atlas_file):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Test file and anatomical labels we want to evaluate
    atlas_vol = load_volfile(atlas_file)
    atlas_vol = atlas_vol[np.newaxis, ..., np.newaxis]

    test_file = open('data/test_example.txt')
    test_strings = test_file.readlines()
    test_strings = [x.strip() for x in test_strings]

    # Set up model
    criterion = LossFunction_mpr_MIND().cuda()
    model = MPR_net_Tr(criterion)
    model = model.cuda()
    model.eval()
    print_network(model)
    model.load_state_dict(torch.load(init_model_file, map_location='cuda:0'))


    # set up atlas tensor
    input_fixed = torch.from_numpy(atlas_vol).cuda().float()
    input_fixed = input_fixed.permute(0, 4, 1, 2, 3)
    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf = trf.cuda()


    for k in range(0, len(test_strings)):

        vol_name, seg_name = test_strings[k].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        input_moving = torch.from_numpy(X_vol).cuda().float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            warp, flow, flow1, refine_flow1, flow2, refine_flow2 = model(input_moving, input_fixed)

        warp = warp.detach().cpu().numpy()
        warp = nib.Nifti1Image(warp[0, 0, :, :, :], np.eye(4))
        nib.save(warp, 'data/res-warped.nii.gz')

        flow = flow.permute(0, 2, 3, 4, 1)
        flow = flow.detach().cpu().numpy()
        flow = nib.Nifti1Image(flow[0, :, :, :, :], np.eye(4))
        nib.save(flow, 'data/res-flow.nii.gz')



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--atlas_file", type=str, dest="atlas_file", default='data/subject-4-T1.nii.gz')
    parser.add_argument("--init_model_file",
                        type=str,
                        dest="init_model_file",
                        default='model/MPR-T2-to-T1atlas/500.ckpt',
                        help="model weight file")
    test(**vars(parser.parse_args()))

