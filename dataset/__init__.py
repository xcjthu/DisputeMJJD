from dataset.JsonFromFiles import JsonFromFilesDataset
from .ParaDataset import ParaDataset
from .DenoiseDataset import DenoiseDataset
from .SecondlevelDataset import SecondlevelDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "ParaBert": ParaDataset,
    "Denoise": DenoiseDataset,
    "SecondlevelBert": SecondlevelDataset,
}
