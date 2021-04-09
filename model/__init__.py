from .model.CNN import TextCNN
from .model.Bert import Bert
from .model.ParaBert import ParaBert
from .model.DenoiseBert import DenoiseBert
from .model.SecondlevelBert import SecondlevelBert

model_list = {
    "CNN": TextCNN,
    "BERT": Bert,
    "ParaBert": ParaBert,
    "Denoise": DenoiseBert,
    "SecondlevelBert": SecondlevelBert,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
