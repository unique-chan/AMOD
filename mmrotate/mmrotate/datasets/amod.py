from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd

import mmcv
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmdet.datasets.custom import CustomDataset
from mmrotate.core import eval_rbbox_map, poly2obb_np


@ROTATED_DATASETS.register_module()
class AMODDataset(CustomDataset):
    CLASSES_PALETTE_COMBINATION_DIC = {
        'Armored': (244, 67, 54),
        'Artillery': (255, 51, 204),
        'Boat': (156, 39, 176),
        'Helicopter': (103, 58, 183),
        'LCU': (63, 81, 181),
        'MLRS': (33, 150, 243),
        'Plane': (0, 188, 212),
        'RADAR': (0, 150, 136),
        'SAM': (76, 175, 80),
        'Self-propelled Artillery': (139, 195, 74),
        'Support': (205, 220, 57),
        'Tank': (255, 122, 0),
        'TEL': (121, 85, 72),
    }

    CLASSES = tuple(CLASSES_PALETTE_COMBINATION_DIC.keys())
    PALETTE = tuple(CLASSES_PALETTE_COMBINATION_DIC.values())

    def __init__(self,
                 ann_file: str,
                 pipeline: Optional[List[dict]],
                 version: str = 'le90',
                 angles: Optional[List[int]] = None,
                 **kwargs) -> None:
        """
        Args:
            ann_file: "directory path" for annotation files (⚠️ ann_file should be ended with '/'!)
             👉 e.g. 'data/split_1024_dota1_0/trainval/annfiles/'
            pipeline: list of data pre-processing strategies
             👉 e.g. [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True), ...]
            version: representation format of oriented bounding boxes (compatible with your detection models)
             👉 e.g. 'le90' or 'le135' or 'oc'
            angles: list of angles (look_angles) used during training/test in our AMOD dataset
             👉 e.g. [0, 10, 20, 30, 40, 50], which means using all look angles available in AMOD
            **kwargs: a syntax in Python that allows a function to accept an arbitrary number of keyword arguments
             👉 These extra keyword arguments are collected into a dictionary within the function
        """
        if angles is None:
            angles = [0,]

        print(f'⭐ [AMOD] Initializing with angles: {angles}')
        self.version = version
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.angles = angles

        super(AMODDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        sample_idx_list = mmcv.list_from_file(ann_file)
        data_info_list = []

        for sample_idx in sample_idx_list:
            for angle in self.angles:
                try:
                    annot_df = pd.read_csv(
                        f'{self.img_prefix}/{sample_idx}/{angle}/ANNOTATION-EO_{sample_idx}_{angle}.csv'
                    ).query('usable == "T"')

                    if not len(annot_df):
                        continue

                    polygons = annot_df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values
                    obb_bboxes = []
                    for polygon in polygons:
                        obb_bbox = poly2obb_np(polygon, self.version)
                        if obb_bbox is not None:
                            obb_bboxes.append(obb_bbox)

                    obb_bboxes = np.array(obb_bboxes, dtype=np.float32)
                    labels = np.array(list(map(lambda label: self.cat2label.get(label, -1),
                                               annot_df['main_class'])), dtype=np.int64)

                    valid_labels_inds = labels != -1
                    labels = labels[valid_labels_inds]
                    obb_bboxes = obb_bboxes[valid_labels_inds]

                    data_info_list.append({
                        'filename': f'{sample_idx}/{angle}/EO_{sample_idx}_{angle}.png',
                        'width': 1920, 'height': 1440,
                        'ann': {
                            'bboxes': obb_bboxes,
                            'labels': labels,
                        }
                    })

                except Exception as e:
                    print(f"Error processing {sample_idx} at angle {angle}: {e}")
                    continue

        return data_info_list

    def _filter_imgs(self):
        """
        Filter out images without valid annotations
        """
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['bboxes'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 use_07_metric=True,
                 nproc=4
    ):
        """
        Evaluate dataset performance with mAP.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr

        if metric == 'mAP':
            assert isinstance(iou_thrs, list)

            mean_aps = []
            for iou_thr in iou_thrs:
                mmcv.print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')

                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    iou_thr=0.5,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)

                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
