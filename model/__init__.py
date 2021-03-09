from .model.CNN import TextCNN
from .model.Bert import Bert
from .model.ParaBert import ParaBert
from .model.DenoiseBert import DenoiseBert

model_list = {
    "CNN": TextCNN,
    "BERT": Bert,
    "ParaBert": ParaBert,
    "Denoise": DenoiseBert,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
