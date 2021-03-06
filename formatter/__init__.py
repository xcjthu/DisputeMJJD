import logging

from .Basic import BasicFormatter
from .CharFormatter import CharFormatter
from .BertFormatter import BertFormatter
from .ParaBertFormatter import ParaBertFormatter
from .DenoiseBertFormatter import DenoiseBertFormatter
from .HierarchyFormatter import HierarchyFormatter
from .LawformerFormatter import LawformerFormatter
from .ParaBertPosFormatter import ParaBertPosFormatter
logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "Char": CharFormatter,
    "BERT": BertFormatter,
    "ParaBert": ParaBertFormatter,
    'Denoise': DenoiseBertFormatter,
    "Hierarchy": HierarchyFormatter,
    "Lawformer": LawformerFormatter,
    "ParaPos": ParaBertPosFormatter,
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
