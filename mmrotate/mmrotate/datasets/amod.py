from collections import OrderedDict
from itertools import chain
from typing import List, Optional

import numpy as np
import pandas as pd

import mmcv
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmdet.datasets.custom import CustomDataset
from mmrotate.core import eval_rbbox_map, poly2obb_np


@ROTATED_DATASETS.register_module()
class AMODDataset(CustomDataset): # Add to __iniy__.py!
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
                 modality: str = 'eo',
                 version: str = 'le90',
                 width: int = 1920,
                 height: int = 1440,
                 ext: str = 'png',
                 angles: Optional[List[int]] = None,
                 **kwargs) -> None:
        """
        Args:
            ann_file: "directory path" for annotation files (⚠️ ann_file should be ended with '/'!)
             👉 e.g. 'data/split_1024_dota1_0/trainval/annfiles/'
            pipeline: list of data pre-processing strategies
             👉 e.g. [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True), ...]
            modality: 'eo' or 'ir'
             👉 e.g. 'eo' or 'ir'
            version: representation format of oriented bounding boxes (compatible with your detection models)
             👉 e.g. 'le90' or 'le135' or 'oc'
            ext: image file extension
             👉 e.g. 'png' or 'jpg'
            angles: list of angles (look_angles) used during training/test in our AMOD dataset
             👉 e.g. [0, 10, 20, 30, 40, 50], which means using all look angles available in AMOD
            **kwargs: a syntax in Python that allows a function to accept an arbitrary number of keyword arguments
             👉 These extra keyword arguments are collected into a dictionary within the function
        """
        if angles is None:
            angles = [0, ]
        if '\\' in kwargs['data_root']:                                     # if spacing is needed
            kwargs['data_root'] = kwargs['data_root'].replace('\\', ' ')
        print(f'▶️ [{self.__class__.__name__}] Initializing with angles: {angles}, '
              f'ann_file: {ann_file}, kwargs: {kwargs}')
        self.version = version
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.angles = angles
        self.modality = modality.upper()
        self.width, self.height = width, height
        self.ext = ext

        super(AMODDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        sample_idx_list = mmcv.list_from_file(ann_file)
        data_info_list = []

        for sample_idx in sample_idx_list:
            for angle in self.angles:
                try:
                    annot_df = pd.read_csv(
                        f'{self.img_prefix}/{sample_idx}/{angle}/ANNOTATION-{self.modality}_{sample_idx}_{angle}.csv'
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
                        'filename': f'{sample_idx}/{angle}/{self.modality}_{sample_idx}_{angle}.{self.ext}',
                        'width': self.width, 'height': self.height,
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
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc,
                    scale_ranges=scale_ranges)


                if isinstance(mean_ap, list):  #  mean_ap가 리스트일 경우
                    for idx, ap in enumerate(mean_ap):  
                        eval_results[f'AP{int(iou_thr * 100):02d}_scale{idx}'] = ap  # 개별 스케일별 AP 저장
                else:
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = mean_ap

                mean_aps.append(mean_ap) 
                print(mean_aps)

            # mean_aps가 2D 리스트인지 확인 후 평탄화
            mean_aps = list(chain(*mean_aps)) if any(isinstance(i, list) for i in mean_aps) else mean_aps

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps) if mean_aps else 0.0
            eval_results.move_to_end('mAP', last=False)

        elif metric == 'recall':
            raise NotImplementedError

        return eval_results



@ROTATED_DATASETS.register_module()
class AMODFineGrainedDataset(AMODDataset):  # Add to __iniy__.py!
    CLASSES_PALETTE_COMBINATION_DIC = {
        'Armored+APC': (244, 67, 54),
        'Armored+IFV': (234, 57, 44),
        'Armored+MRAP': (229, 52, 39),
        'Artillery+Artillery': (255, 51, 204),
        'Boat+Boat': (156, 39, 176),
        'Boat+RHIB': (151, 34, 171),
        'Helicopter+AH': (103, 58, 183),
        'Helicopter+CH': (98, 53, 178),
        'Helicopter+OH': (93, 48, 173),
        'Helicopter+UH': (88, 43, 168),
        'LCU+LCU': (63, 81, 181),
        'MLRS+MLRS': (33, 150, 243),
        'Plane+Attacker': (0, 188, 212),
        'Plane+Bomber': (0, 183, 207),
        'Plane+Cargo': (0, 178, 202),
        'Plane+Fighter': (0, 173, 197),
        'RADAR+RADAR': (0, 150, 136),
        'SAM+SAM': (76, 175, 80),
        'Self-propelled Artillery+Self-propelled Artillery': (139, 195, 74),
        'Support+Mil_car': (205, 220, 57),
        'Support+Mil_truck': (200, 215, 52),
        'Support+ASV': (195, 210, 47),
        'Tank+Tank': (255, 122, 0),
        'TEL+TEL': (121, 85, 72),
    }
    CLASSES = tuple(CLASSES_PALETTE_COMBINATION_DIC.keys())
    PALETTE = tuple(CLASSES_PALETTE_COMBINATION_DIC.values())

    def __init__(self, ann_file, **kwargs):
        super().__init__(ann_file, **kwargs)

    def load_annotations(self, ann_file):
        sample_idx_list = mmcv.list_from_file(ann_file)
        data_info_list = []

        for sample_idx in sample_idx_list:
            for angle in self.angles:
                try:
                    annot_df = pd.read_csv(
                        f'{self.img_prefix}/{sample_idx}/{angle}/ANNOTATION-{self.modality}_{sample_idx}_{angle}.csv'
                    ).query('usable == "T"')

                    if not len(annot_df):
                        continue

                    annot_df['middle_class'] = annot_df['main_class'] + '+' + annot_df['middle_class']

                    polygons = annot_df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values
                    obb_bboxes = []
                    for polygon in polygons:
                        obb_bbox = poly2obb_np(polygon, self.version)
                        if obb_bbox is not None:
                            obb_bboxes.append(obb_bbox)

                    obb_bboxes = np.array(obb_bboxes, dtype=np.float32)
                    labels = np.array(list(map(lambda label: self.cat2label.get(label, -1),
                                               annot_df['middle_class'])), dtype=np.int64)

                    valid_labels_inds = labels != -1
                    labels = labels[valid_labels_inds]
                    obb_bboxes = obb_bboxes[valid_labels_inds]

                    data_info_list.append({
                        'filename': f'{sample_idx}/{angle}/{self.modality}_{sample_idx}_{angle}.{self.ext}',
                        'width': self.width, 'height': self.height,
                        'ann': {
                            'bboxes': obb_bboxes,
                            'labels': labels,
                        }
                    })

                except Exception as e:
                    print(f"Error processing {sample_idx} at angle {angle}: {e}")
                    continue

        return data_info_list


@ROTATED_DATASETS.register_module()
class AMODwithCivilianDataset(AMODDataset):  # Add to __iniy__.py!
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
        'civilian': (0, 0, 0),
    }
    CLASSES = tuple(CLASSES_PALETTE_COMBINATION_DIC.keys())
    PALETTE = tuple(CLASSES_PALETTE_COMBINATION_DIC.values())

    def __init__(self, ann_file, **kwargs):
        super().__init__(ann_file, **kwargs)


@ROTATED_DATASETS.register_module()
class AMODwithCivilianFineGrainedDataset(AMODFineGrainedDataset):  # Add to __iniy__.py!
    CLASSES_PALETTE_COMBINATION_DIC = {
        'Armored+APC': (244, 67, 54),
        'Armored+IFV': (234, 57, 44),
        'Armored+MRAP': (229, 52, 39),
        'Artillery+Artillery': (255, 51, 204),
        'Boat+Boat': (156, 39, 176),
        'Boat+RHIB': (151, 34, 171),
        'Helicopter+AH': (103, 58, 183),
        'Helicopter+CH': (98, 53, 178),
        'Helicopter+OH': (93, 48, 173),
        'Helicopter+UH': (88, 43, 168),
        'LCU+LCU': (63, 81, 181),
        'MLRS+MLRS': (33, 150, 243),
        'Plane+Attacker': (0, 188, 212),
        'Plane+Bomber': (0, 183, 207),
        'Plane+Cargo': (0, 178, 202),
        'Plane+Fighter': (0, 173, 197),
        'RADAR+RADAR': (0, 150, 136),
        'SAM+SAM': (76, 175, 80),
        'Self-propelled Artillery+Self-propelled Artillery': (139, 195, 74),
        'Support+Mil_car': (205, 220, 57),
        'Support+Mil_truck': (200, 215, 52),
        'Support+ASV': (195, 210, 47),
        'Tank+Tank': (255, 122, 0),
        'TEL+TEL': (121, 85, 72),
        'civilian': (0, 0, 0),
    }
    CLASSES = tuple(CLASSES_PALETTE_COMBINATION_DIC.keys())
    PALETTE = tuple(CLASSES_PALETTE_COMBINATION_DIC.values())

    def __init__(self, ann_file, **kwargs):
        super().__init__(ann_file, **kwargs)