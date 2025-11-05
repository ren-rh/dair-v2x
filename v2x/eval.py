import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from v2x_utils import range2box, id_to_str, Evaluator
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel

def vis_pred_label_2(pred, label, str, out_dir):
        truth = []
        predict = []        
        # 创建保存可视化结果的文件夹路径
        img_save_floder_path = osp.join(out_dir,'vis_results_dair','BEV_pred_label')
        if not osp.exists(img_save_floder_path):
            os.makedirs(img_save_floder_path)
        # 清除当前图形    
        plt.cla()
        # 绘制车辆在全局坐标系中的位置，以蓝色圆点表示
        # plt.scatter(-veh_c_s_world[1,0],veh_c_s_world[0,0],c='b',marker='o')
        # 遍历每个标签（真值）并绘制边界框
        for i in range(label.shape[0]):
            x3 = label[i,[0,1,2,3,0],0]  
            y3 = label[i,[0,1,2,3,0],1]  
            plt.plot(-y3,x3,'g')
            #truth.append((-y3, x3))
            for j in range(5):
                truth.append((-y3[j], x3[j]))
            # plt.scatter(label[i,3,0],label[i,3,1],c='g',marker='o')
            # plt.axis('equal')   
        # 遍历每个预测结果并绘制边界框    
        for i in range(pred.shape[0]):
            x = pred[i,[0,4,7,3,0],0]  
            y = pred[i,[0,4,7,3,0],1]  
            plt.plot(-y,x,'r')
            #predict.append((-y, x))
            for j in range(5):
                predict.append((-y[j], x[j]))
            # plt.scatter(pred[i,0,0],pred[i,0,1],c='r',marker='*')
        
        plt.axis('equal')
        # 比例
        #plt.ylim([-5, 65])
        #plt.xlim([-40, 30])

        img_save_path = osp.join(img_save_floder_path,str+'pred_label.png')
        plt.savefig(img_save_path,dpi=300)
        
        if not osp.exists(os.path.join(out_dir, 'vis_results_dair', 'truth')):
            os.makedirs(os.path.join(out_dir, 'vis_results_dair', 'truth'))
        if not osp.exists(os.path.join(out_dir, 'vis_results_dair', 'predict')):
            os.makedirs(os.path.join(out_dir, 'vis_results_dair', 'predict'))
        t_path = os.path.join(out_dir, 'vis_results_dair', 'truth', str+'.txt')
        p_path = os.path.join(out_dir, 'vis_results_dair', 'predict', str+'.txt')
        count = 0
        with open(t_path, 'w') as file:
            for item in truth:
                file.write(f"{item}")
                count += 1
                if count % 5 == 0:
                    file.write("\n")
        count = 0
        with open(p_path, 'w') as file:
            for item in predict:
                file.write(f"{item}")
                count += 1
                if count % 5 == 0:
                    file.write("\n")
  

def eval_vic(args, dataset, model, evaluator):
    idx = -1
    for VICFrame, label, filt in tqdm(dataset):
        idx += 1
        # if idx % 10 != 0:
        #     continue
        if 'spd' in args.dataset:
            veh_id = VICFrame.vehicle_frame().get("frame_id")
        else:
            try:
                veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
            except Exception:
                veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")

        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )

        evaluator.add_frame(pred, label)
        pipe.flush()
        pred["label"] = label["boxes_3d"]
        pred["veh_id"] = veh_id
        #save_pkl(pred, osp.join(args.output, "result", pred["veh_id"] + ".pkl"))
        
        pp = pred['boxes_3d']
        ll = label['boxes_3d']
        #veh_id_str = "%06d" % veh_id
        out_dir = "result/"
        vis_pred_label_2(pp,ll,veh_id+'_0',out_dir)
        

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model, evaluator):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)
        if args.sensortype == "camera":
            evaluator.add_frame(pred, label["camera"])
        elif args.sensortype == "lidar":
            evaluator.add_frame(pred, label["lidar"])
        save_pkl({"boxes_3d": label["lidar"]["boxes_3d"]}, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
        val_data_path=args.val_data_path
    )

    logger.info("loading evaluator")
    evaluator = Evaluator(args.pred_classes)

    logger.info("loading model")
    print("single", args.eval_single)  #False
    print("model", args.model)  #late_fusion
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        ### Patch for FFNet evaluation ###
        if args.model =='feature_flow':
            model.model.data_root = args.input
            model.model.test_mode = args.test_mode
        #############################
        eval_vic(args, dataset, model, evaluator)
