from dataset.JsonFromFiles import JsonFromFilesDataset
from .ParaDataset import ParaDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "ParaBert": ParaDataset,
}
