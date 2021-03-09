from dataset.JsonFromFiles import JsonFromFilesDataset
from .ParaDataset import ParaDataset
from .DenoiseDataset import DenoiseDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "ParaBert": ParaDataset,
    "Denoise": DenoiseDataset,
}
