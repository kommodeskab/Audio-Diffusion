from src.networks.pretrained import PretrainedNetwork
import torch

if __name__ == "__main__":
    model = PretrainedNetwork(project_name="example", experiment_id="251124000407", key="encoder")
    input = torch.randn(1, 1, 32, 32)
    output = model(input)
    print(output)