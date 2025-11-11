from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Datasets(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'CNV', 'DME'),
        palette=[[0, 0, 0], [0, 128, 128], [128, 128, 0]])

    def __init__(self,
                 img_suffix='.jpeg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)