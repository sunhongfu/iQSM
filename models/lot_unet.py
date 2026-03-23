import torch.nn as nn

from .unet import Unet  # noqa: F401 (re-exported for convenience)


class LoT_Unet(nn.Module):
    def __init__(self, LoT_Layer, Unet_part):
        super(LoT_Unet, self).__init__()
        self.Unet = Unet_part
        self.LoT_Layer = LoT_Layer

    def forward(self, wphs, masks, TEs, B0):
        LoT_Filtered_results, LearnableFilterd_results = self.LoT_Layer(wphs, masks, TEs, B0)
        recon = self.Unet(LoT_Filtered_results, LearnableFilterd_results)
        recon = recon / 4  # simple linear normalisation due to training settings
        return recon
