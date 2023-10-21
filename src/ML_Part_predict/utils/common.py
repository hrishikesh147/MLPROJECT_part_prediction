from pathlib import Path
import yaml
from box import ConfigBox
from src.ML_Part_predict import logger
from ensure import ensure_annotations
import os,sys
import json

@ensure_annotations
def read_yaml(filepath:Path) -> ConfigBox:
    try:
        with open(filepath) as y:
            content=yaml.safe_load(y)
            logger.info(f"yaml file {content} loaded successfully")
            return ConfigBox(content)
        
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(filep:list, verbose=True):
    for i in filep:
        os.makedirs(i,exist_ok=True)
        if verbose:
            logger.info(f"created directory at path {i}")

@ensure_annotations
def save_json(save_in_path:Path,data:dict):
    with open(save_in_path,"w") as f:
        json.dump(data,f,indent=4)
    logger.info(f"Json file saved at {save_in_path}")


@ensure_annotations
def load_json(load_from_path:Path):
    with open(load_from_path,"r") as f:
        content=json.load(f)
    logger.info(f"json file {load_from_path} loaded")
    return ConfigBox(content)

@ensure_annotations
def get_size(path:Path) -> str:
    size_kb=round(os.path.getsize(path)/1024)
    logger.info(f"size of {path} is {size_kb}")
    return f"{size_kb} KB"
