from dataset.JsonFromFiles import JsonFromFilesDataset
from .ParaDataset import ParaDataset
from .DenoiseDataset import DenoiseDataset
from .ParaDatasetPos import ParaPosDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "ParaBert": ParaDataset,
    "Denoise": DenoiseDataset,
    "ParaPos": ParaPosDataset
}
