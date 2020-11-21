# Copyright 2020 Toyota Research Institute.  All rights reserved.
# This script provides a demo inference a model trained on Cityscapes dataset.
import warnings
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection.image_list import ImageList

from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image,visualize_detection_image
from realtime_panoptic.utils.onnx_utls import convert_group_norm_to_onnxable, convert_frozen_batchnorm_to_batchnorm

def export_model(cfg, weight_file, model_file, image_width, image_height):
    # Build original model and load checkpoint
    # Initialize model.
    model = RTPanoNet(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.panoptic.num_classes,
        things_num_classes=cfg.model.panoptic.num_thing_classes,
        pre_nms_thresh=cfg.model.panoptic.pre_nms_thresh,
        pre_nms_top_n=cfg.model.panoptic.pre_nms_top_n,
        nms_thresh=cfg.model.panoptic.nms_thresh,
        fpn_post_nms_top_n=cfg.model.panoptic.fpn_post_nms_top_n,
        instance_id_range=cfg.model.panoptic.instance_id_range)
    model = convert_group_norm_to_onnxable(model)
    model = convert_frozen_batchnorm_to_batchnorm(model)

    model.load_state_dict(torch.load(weight_file))
    model.eval()
    img = torch.randn(1, 3, image_height, image_width)
    torch.onnx.export(model, img, model_file, verbose=False, do_constant_folding=True)

    # Try to load model and check validity
    onnx_model = onnx.load(model_file)

    onnx.checker.check_model(onnx_model)

    # Now we do a check between outputs
    # Run unrolled model
    with torch.no_grad():
        boxes, scores, semantic_logits, levelness_logits = model(img)

        boxes, scores = boxes.numpy().squeeze(), scores.numpy().squeeze()

    # Run ONNX model
    sess = rt.InferenceSession(model_file)
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: img.numpy()})
    boxes_onnx, scores_onnx = pred_onnx[0].squeeze(), pred_onnx[1].squeeze()
    #
    # # Now compare output tensors
    assert np.allclose(boxes, boxes_onnx, atol=1e-03)
    assert np.allclose(scores, scores_onnx, atol=1e-03)

def demo():
    # Parse the input arguments.
    parser = argparse.ArgumentParser(description="Simple demo for real-time-panoptic model")
    parser.add_argument("--config-file", metavar="FILE", help="path to config", required=True)
    parser.add_argument("--pretrained-weight", metavar="FILE", help="path to pretrained_weight", required=True)
    parser.add_argument("--image_width", type=int, default=2048, help="ONNX input image width")
    parser.add_argument("--image_height", type=int, default=1024, help="ONNX input image height")

    parser.add_argument("opts", help="Modify config options via CLI", default=None, nargs=argparse.REMAINDER)
    # parser.add_argument("--device", help="inference device", default='cuda')
    args = parser.parse_args()

    # General config object from given config files.
    cfg.merge_from_file(args.config_file)

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    print("\n" + collect_env_info())

    # export model
    export_model(cfg, weight_file, args.onnx_file, args.image_width, args.image_height)

    # Initialize model.
    # model = RTPanoNet(
    #     backbone=cfg.model.backbone,
    #     num_classes=cfg.model.panoptic.num_classes,
    #     things_num_classes=cfg.model.panoptic.num_thing_classes,
    #     pre_nms_thresh=cfg.model.panoptic.pre_nms_thresh,
    #     pre_nms_top_n=cfg.model.panoptic.pre_nms_top_n,
    #     nms_thresh=cfg.model.panoptic.nms_thresh,
    #     fpn_post_nms_top_n=cfg.model.panoptic.fpn_post_nms_top_n,
    #     instance_id_range=cfg.model.panoptic.instance_id_range)
    # device = args.device
    # model.to(device)
    # model.load_state_dict(torch.load(args.pretrained_weight))

    # # Print out mode architecture for sanity checking.
    # print(model)

    # # Prepare for model inference.
    # model.eval()
    # input_image = Image.open(args.input)
    # data = {'image': input_image}
    # # data pre-processing
    # normalize_transform = P.Normalize(mean=cfg.input.pixel_mean, std=cfg.input.pixel_std, to_bgr255=cfg.input.to_bgr255)
    # transform = P.Compose([
    #     P.ToTensor(),
    #     normalize_transform,
    # ])
    # data = transform(data)
    # print("Done with data preparation and model configuration.")
    # with torch.no_grad():
    #     input_image_list = ImageList([data['image'].to(device)], image_sizes=[input_image.size[::-1]])
    #     panoptic_result, _ = model.forward(input_image_list)
    #     print("Done with model inference.")
    #     print("Process and visualizing the outputs...")
    #     instance_detection = [o.to('cpu') for o in panoptic_result["instance_segmentation_result"]]
    #     semseg_logics = [o.to('cpu') for o in panoptic_result["semantic_segmentation_result"]]
    #     semseg_prob = [torch.argmax(semantic_logit , dim=0) for semantic_logit in  semseg_logics]

    #     seg_vis = visualize_segmentation_image(semseg_prob[0], input_image, cityscapes_colormap)
    #     Image.fromarray(seg_vis.astype('uint8')).save('semantic_segmentation_result.jpg')
    #     print("Saved semantic segmentation visualization in semantic_segmentation_result.jpg")
    #     det_vis = visualize_detection_image(instance_detection[0], input_image, cityscapes_instance_label_name)
    #     Image.fromarray(det_vis.astype('uint8')).save('instance_segmentation_result.jpg')
    #     print("Saved instance segmentation visualization in instance_segmentation_result.jpg")
    #     print("Demo finished.")

if __name__ == "__main__":
    demo()