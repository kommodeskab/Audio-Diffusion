from diffusers import UNet2DModel, UNet1DModel
import torch

class UNet2D(UNet2DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
        
class PretrainedUNet2D(UNet2D):
    def __init__(
        self,
        model_id : str,
        **kwargs,
    ):
        super().__init__()
        dummy_model : torch.nn.Module = UNet2DModel.from_pretrained(pretrained_model_name_or_path = model_id, **kwargs)
        self.__dict__ = dummy_model.__dict__.copy()
        self.load_state_dict(dummy_model.state_dict())
    
class UNet1D(UNet1DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
    
class UNet1DWithConv(UNet1D):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        **kwargs,
    ):
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            **kwargs,
        )
        
        self.downconv = torch.nn.Conv1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size // 2,
        )
        self.upconv = torch.nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size // 2,
        )
        
    def forward(self, x : torch.Tensor, time_step : torch.Tensor) -> torch.Tensor:
        original_length = x.shape[-1]
        x = self.downconv(x)
        x = super().forward(x, time_step)
        x = self.upconv(x)
        padding = original_length - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padding))
        return x
    
if __name__ == "__main__":
    x = torch.randn(1, 1, 32000)
    timestep = torch.randint(0, 100, (1,))
    
    model_1 = UNet1DWithConv(in_channels = 1, out_channels = 1, kernel_size=13, stride=4, extra_in_channels=16)
    y = model_1(x, timestep)
    print(y.shape)
    print(y)