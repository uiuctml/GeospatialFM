# from GeospatialFM.models.wrappers.satmae_wrapper import SatMAEEncoder, SatMAEConfig
from GeospatialFM.models.spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder, SpatialSpectralLowRankViTConfig
from GeospatialFM.models.wrappers.specvit_wrapper import SpecViTEncoder, SpecViTConfig
from GeospatialFM.models.wrappers.dinov3_wrapper import DINOv3Encoder, DINOv3Config
from GeospatialFM.models.wrappers.dofa_wrapper import DOFAEncoder, DOFAConfig
from GeospatialFM.models.wrappers.spatsigma_wrapper import SpatSigmaClsEncoder, SpatSigmaSegEncoder, SpatSigmaConfig

ENCODER_CONFIGS = {
    "lessvit": SpatialSpectralLowRankViTConfig,
    # "satmae": SatMAEConfig,
    "specvit": SpecViTConfig,
    "dinov3": DINOv3Config,
    "dofa": DOFAConfig,
    "spatsigma": SpatSigmaConfig,
}

ENCODER_MODELS = {
    "lessvit": SpatialSpectralLowRankViTEncoder,
    # "satmae": SatMAEEncoder,
    "specvit": SpecViTEncoder,
    "dinov3": DINOv3Encoder,
    "dofa": DOFAEncoder,
    "spatsigma_cls": SpatSigmaClsEncoder,
    "spatsigma_seg": SpatSigmaSegEncoder,
}