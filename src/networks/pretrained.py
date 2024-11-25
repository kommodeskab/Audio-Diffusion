import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
import glob
from src.utils import get_ckpt_path

class PretrainedNetwork(nn.Module):
    def __init__(
        self,
        project_name : str,
        experiment_id : str,
        key : str,
    ):
        """
        Load a pretrained network from a wandb run.
        The checkpoint to the Pytorch Lightning model needs to be stored locally. 
        The checkpoint is loaded and the weights are copied to the current network.
        """
        super().__init__()
        # instantiate a dummy network to get the methods
        # information about the network is stored in the wandb config
        api = wandb.Api()
        run = api.run(f"{project_name}/{experiment_id}")
        config = run.config
        network_cfg = config['model'][key]
        dummy_network : torch.nn.Module = instantiate(network_cfg)
        
        # copy the methods from the dummy network
        for name in dir(dummy_network):
            method = getattr(dummy_network, name)
            if callable(method):
                setattr(self, name, method)
                
        for name, module in dummy_network.named_children():
            self.add_module(name, module)
        
        # load the weights from the checkpoint
        ckpt_path = get_ckpt_path(project_name, experiment_id)
        state_dict : dict[str, torch.Tensor] = torch.load(ckpt_path, weights_only=True)['state_dict']
        state_dict = {k[len(key) + 1:] : v for k, v in state_dict.items() if k.startswith(key)}
        self.load_state_dict(state_dict)
        print(f"Loaded a pretrained network from {ckpt_path}")