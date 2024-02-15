import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.livox.ld_base_v1 import LD_base

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import copy
import sensor_msgs.point_cloud2 as pc2



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


class ros_demo():
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.starter, self.ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)

        self.offset_angle = 0
        self.offset_ground = 1.8
        self.point_cloud_range = [0, -44.8, -2, 224, 44.8, 4]

    def mask_points_out_of_range(self, pc, pc_range):
        pc_range = np.array(pc_range)
        pc_range[3:6] -= 0.01  #np -> cuda .999999 = 1.0
        mask_x = (pc[:, 0] > pc_range[0]) & (pc[:, 0] < pc_range[3])
        mask_y = (pc[:, 1] > pc_range[1]) & (pc[:, 1] < pc_range[4])
        mask_z = (pc[:, 2] > pc_range[2]) & (pc[:, 2] < pc_range[5])
        mask = mask_x & mask_y & mask_z
        pc = pc[mask]
        return pc

    def receive_from_ros(self, msg):
       
        # points = np.frombuffer(msg.data, dtype=np.float32)
        # points = points.reshape(-1, 6)
        # points = points[:, :5]
        pc_points = list(pc2.read_points(msg, field_names=(
            "x", "y", "z", "intensity"), skip_nans=True))
        points = np.array(pc_points, dtype=np.float32)
        

        points_list = np.copy(points)

        # preprocess 
        # points_list[:, 2] += points_list[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
        # rviz_points = copy.deepcopy(points_list)
        # points_list = self.mask_points_out_of_range(points_list, self.point_cloud_range)

        min_val = np.min(points_list[:, 3])
        max_val = np.max(points_list[:, 3])
        points_list[:, 3] = (points_list[:, 3] - min_val) * \
            (1 / (max_val - min_val))

        input_dict = {
                'points': points_list,
                'frame_id': 0
                }

        data_dict = demo_dataset.prepare_data(data_dict=input_dict)

        return data_dict
    
    def online_inference(self, msg):
        with torch.no_grad():
            data_dict = self.receive_from_ros(msg)
            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)
            
            pred_dicts, _ = model.forward(data_dict)

            print(pred_dicts)

            V.draw_scenes(
                points=data_dict['points'][:,
                                        1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

if __name__ == '__main__':
    rospy.init_node('det3d')
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    # These are used for livox models from https://github.com/Livox-SDK/livox_detection
    # Instead of the build network below

    # model = LD_base()
    # checkpoint = torch.load(args.ckpt, map_location=torch.device('cuda:0'))
    # model.load_state_dict({k.replace('module.', ''): v for k,
    #                       v in checkpoint['model_state_dict'].items()})
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    demo_ros = ros_demo(model, args)
    sub = rospy.Subscriber(
        "/livox/lidar", PointCloud2, queue_size=10, callback=demo_ros.online_inference)
    print("set up subscriber!")

    rospy.spin()
