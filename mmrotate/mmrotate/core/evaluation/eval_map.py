# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    det_bboxes = np.array(det_bboxes)  # 탐지된 바운딩 박스
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=bool), 
         np.ones(gt_bboxes_ignore.shape[0], dtype=bool))
    )  # GT 및 무시해야 할 GT 리스트 생성

    # 스택하여 GT와 무시할 GT를 합침
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))  

    num_dets = det_bboxes.shape[0]   # 탐지된 바운딩 박스 개수
    num_gts = gt_bboxes.shape[0]     # GT 바운딩 박스 개수

    if area_ranges is None:  # 평가할 바운딩 박스 크기 제한이 없는 경우
        area_ranges = [(None, None)]

    num_scales = len(area_ranges)  # 평가할 크기 범위 개수
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # GT가 없는 경우 모든 탐지된 바운딩 박스를 False Positive로 처리
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    # IOU 계산
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    
    ious_max = ious.max(axis=1)  # 각 탐지된 바운딩 박스에 대해 최대 IoU 값
    ious_argmax = ious.argmax(axis=1)  # 최대 IoU를 가지는 GT의 인덱스
    sort_inds = np.argsort(-det_bboxes[:, -1])  # 신뢰도 점수에 따라 정렬

    # GT의 크기 계산 (가로 * 세로)
    gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]

    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)  # GT가 탐지되었는지 여부

        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)  # 모든 GT를 포함
        else:
            # ✅ GT 크기에 따라 필터링 적용
            gt_area_ignore = (gt_areas < min_area) | (gt_areas > max_area)  

        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:  # 해당 GT가 아직 매칭되지 않은 경우
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1  # True Positive
                    else:
                        fp[k, i] = 1  # 중복 탐지는 False Positive 처리
            elif min_area is None:
                fp[k, i] = 1  # IoU가 낮고, 크기 조건이 없을 경우 False Positive
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if min_area <= area < max_area:  
                    fp[k, i] = 1  # 크기 범위 내에 속하는 경우 False Positive

    return tp, fp



def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num

    area_ranges = ([(rg[0], rg[1]) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    
    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3] # 넓이 
                
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)