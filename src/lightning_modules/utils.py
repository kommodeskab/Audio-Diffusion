import os

model_dict = {}

def ckpt_path_from_id(project_name : str, model_id : str) -> str:
    folder_name = f"logs/{project_name}/{model_dict[model_id]}/checkpoints"
    return os.path.join(folder_name, os.listdir(folder_name)[0])