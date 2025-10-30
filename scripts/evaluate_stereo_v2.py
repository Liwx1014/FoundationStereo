# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys
import argparse
import logging
import imageio.v2 as imageio
import numpy as np
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import toOpen3dCloud, depth2xyzmap
from core.foundation_stereo import *

def compute_metrics(gt, pred, bad_pixel_thresholds=[1, 2, 3, 5]):
    if gt.shape != pred.shape:
        h, w = gt.shape
        pred = pred[:h, :w]
    
    mask = (gt > 0) & np.isfinite(gt)
    if not np.any(mask):
        return { 'EPE': np.nan, 'D1': np.nan, **{f'BP-{t}': np.nan for t in bad_pixel_thresholds} }

    error = np.abs(gt[mask] - pred[mask])
    epe = np.mean(error)
    d1_mask = (error > 3) & (error > 0.05 * gt[mask])
    d1 = np.sum(d1_mask) / np.sum(mask) * 100.0
    metrics = {f'BP-{t}': np.sum(error > t) / np.sum(mask) * 100.0 for t in bad_pixel_thresholds}
    metrics['EPE'] = epe
    metrics['D1'] = d1
    return metrics

def find_matching_files(dataset_dir):
    left_dir, right_dir, disparity_dir = [os.path.join(dataset_dir, d) for d in ['left', 'right', 'disparity']]
    if not all(os.path.isdir(d) for d in [left_dir, right_dir, disparity_dir]):
        raise FileNotFoundError("数据集目录下必须包含 'left', 'right', 和 'disparity' 子目录")
    #left_files = sorted(glob(os.path.join(left_dir, '*.png'))) or sorted(glob(os.path.join(left_dir, '*.jpg')))
    png_files = glob(os.path.join(left_dir, '*.png'))
    jpg_files = glob(os.path.join(left_dir, '*.jpg'))
    left_files = sorted(png_files + jpg_files)
    file_matches = []
    for left_path in left_files:
        prefix = os.path.splitext(os.path.basename(left_path))[0]
        right_path_cand = glob(os.path.join(right_dir, f'{prefix}.*'))
        disp_path_cand = glob(os.path.join(disparity_dir, f'{prefix}.*')) 
        if right_path_cand and disp_path_cand:
            file_matches.append((prefix, left_path, right_path_cand[0], disp_path_cand[0]))
        else:
            logging.warning(f"无法为 {prefix} 找到匹配的右图或视差图，已跳过。")
    return file_matches

if __name__=="__main__":
    
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir', default=f'{code_dir}/../assets/my_dataset', type=str, help='包含left/right/disparity子目录的数据集路径')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1. For evaluation, this is often 1.0')
  parser.add_argument('--hiera', default=1, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--valid_iters', type=int, default=48, help='number of flow-field updates during forward pass')
  parser.add_argument('--bad_pixel_thresholds', nargs='+', type=int, default=[1, 3, 5], help='BP-X指标的阈值列表')
  parser.add_argument('--disp_scale_factor', type=float, default=256.0, help='PNG视差图的缩放因子 (真实视差 = 像素值 / 缩放因子)')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")

  model = FoundationStereo(args)
  ckpt = torch.load(ckpt_dir)
  model.load_state_dict(ckpt['model'])
  model.cuda()
  model.eval()

  file_matches = find_matching_files(args.dataset_dir)
  results_list = []

  for prefix, left_file, right_file, disp_file in tqdm(file_matches, desc="Evaluating Stereo Model"):
      logging.info(f"--- Processing: {prefix} ---")
      img0, img1 = imageio.imread(left_file), imageio.imread(right_file)
      gt_disp_raw = imageio.imread(disp_file)
      
      if gt_disp_raw.ndim == 3: gt_disp_raw = gt_disp_raw[:,:,0]
      gt_disp = gt_disp_raw.astype(np.float32) / args.disp_scale_factor if args.disp_scale_factor > 0 else gt_disp_raw.astype(np.float32)
      if img0.shape[2] == 4: img0 = img0[:, :, :3]
      if img1.shape[2] == 4: img1 = img1[:, :, :3]
      scale = args.scale
      assert scale <= 1, "scale must be <=1"
      if scale < 1.0:
          logging.info(f"Applying uniform downscaling with scale factor: {scale:.2f}")
          h_orig, w_orig = img0.shape[:2]
          new_h, new_w = int(h_orig * scale), int(w_orig * scale)
          img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_AREA)
          img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
          gt_disp = cv2.resize(gt_disp, (new_w, new_h), interpolation=cv2.INTER_NEAREST) * scale
      
      H, W = img0.shape[:2]
     
      img0_ori = img0.copy()
      logging.info(f"Inference shape: {H}x{W}")

      img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
      img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
      padder = InputPadder(img0.shape, divis_by=32, force_square=False)
      img0, img1 = padder.pad(img0, img1)

      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.reset_peak_memory_stats()
      start_event.record()

      with torch.cuda.amp.autocast(True):
        if not args.hiera:
          logging.info("Running single-scale inference")
          disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
          logging.info("Running hierarchical inference")
          disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

      end_event.record()
      torch.cuda.synchronize()

      inference_time_ms = start_event.elapsed_time(end_event)
      peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
      
      disp = padder.unpad(disp.float())
      pred_disp_raw = disp.data.cpu().numpy().reshape(H,W)
  
      valid_mask = (gt_disp > 0) & np.isfinite(gt_disp)
      
      if np.any(valid_mask):
          # 计算真实视差和原始预测视差的最大值
          gt_max = np.max(gt_disp[valid_mask])
          raw_pred_max = np.max(pred_disp_raw[valid_mask])
          # 检查它们的比率是否接近8
          # (设置一个容忍区间，例如7到9，以防微小误差)
          if gt_max > 0 and (raw_pred_max / gt_max > 7 and raw_pred_max / gt_max < 9):
              logging.info(f"Raw prediction scale (~{raw_pred_max/gt_max:.2f}x) is much larger than GT scale.")
              logging.info(f"Correcting prediction scale by 1/8.")
              pred_disp = pred_disp_raw / 8
          else:
              pred_disp = pred_disp_raw
              logging.info("Raw prediction scale is close to GT scale. No correction applied.")
      else:
          # 如果没有GT，我们无法判断，只能依赖于use_hierarchical的简单逻辑
          logging.warning("No valid GT for dynamic scaling check. Falling back to simple logic.")
          if not args.hiera:
              pred_disp = pred_disp_raw / 8
      

      if np.any(valid_mask):
          # 打印原始预测（校正前）的统计信息，以了解模型的原始输出范围
          raw_pred_max = np.max(pred_disp_raw[valid_mask])
          logging.info(f"Disparity Raw Prediction Stats (on valid GT pixels): Max={raw_pred_max:.2f}")

          # 打印真实视差和校正后预测视差的统计信息
          gt_max = np.max(gt_disp[valid_mask])
          pred_max = np.max(pred_disp[valid_mask])
          gt_mean = np.mean(gt_disp[valid_mask])
          pred_mean = np.mean(pred_disp[valid_mask])
          logging.info(f"Disparity Comparison (on valid GT pixels):")
          logging.info(f"  - Ground Truth: Max={gt_max:.2f}, Mean={gt_mean:.2f}")
          logging.info(f"  - Prediction:   Max={pred_max:.2f}, Mean={pred_mean:.2f} (after dividing by 8)")
      else:
          logging.warning("No valid ground truth pixels found for statistics.")
      metrics = compute_metrics(gt_disp, pred_disp, args.bad_pixel_thresholds)
      
      current_result = {
          'filename': prefix,
          'inference_size': f'{H}x{W}',
          **metrics,
          'inference_time_ms': inference_time_ms,
          'peak_memory_mb': peak_memory_mb
      }
      results_list.append(current_result)
      
      logging.info(f"Metrics for {prefix}: {metrics}")
      Visualize = True
      if Visualize:
        vis_pred = vis_disparity(pred_disp)
        vis_gt = vis_disparity(gt_disp)
        vis = np.concatenate([img0_ori, vis_pred, vis_gt], axis=1)
        os.makedirs(os.path.join(args.out_dir, "visuals"), exist_ok=True)
        imageio.imwrite(os.path.join(args.out_dir, "visuals", f'{prefix}_vis.png'), vis)
      
  if results_list:
      df = pd.DataFrame(results_list)
      avg_metrics = df.mean(numeric_only=True).to_dict()
      avg_metrics.update({'filename': 'Average', 'inference_size': '-'})
      df_avg = pd.DataFrame([avg_metrics])
      df = pd.concat([df, df_avg], ignore_index=True)
      output_csv_path = os.path.join(args.out_dir, 'evaluation_results_fds.csv')
      df.to_csv(output_csv_path, index=False, float_format='%.4f')
      logging.info(f"Evaluation results saved to {output_csv_path}")