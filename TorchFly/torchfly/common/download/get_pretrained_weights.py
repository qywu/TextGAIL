import torch
import torch.nn as nn
import logging

from .download_utils import gdrive_download
from .supported_pretrained_weights_list import supported_models_gdrive_map

logger = logging.getLogger(__name__)


def download_gdrive_id(model_name, file_id):
    url = "https://drive.google.com/uc?id=" + file_id
    filepath = gdrive_download(url, "models", f"{model_name}.pth")
    states_dict = torch.load(filepath)
    return states_dict


def get_pretrained_weights(modelname=None, url=None, gdrive=True):
    # TODO add different sources not only from gdrive
    logger.info(f"Loading {modelname} weights >>")

    if modelname in supported_models_gdrive_map.keys():
        states_dict = download_gdrive_id(
            modelname, supported_models_gdrive_map[modelname])
        return states_dict
    else:
        raise NotImplementedError
