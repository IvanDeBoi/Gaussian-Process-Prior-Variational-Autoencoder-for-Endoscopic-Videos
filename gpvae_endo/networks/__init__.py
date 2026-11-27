"""Neural networks of the model: encoders, decoders, discriminators and the segmentation net."""
from gpvae_endo.networks.cnn import CNNDecoder, CNNDecoderResidual, CNNEncoder
from gpvae_endo.networks.endovae import EndoVAEDecoder, EndoVAEEncoder
from gpvae_endo.networks.vit import GastronetViTEncoder, GastronetViTEncoderLoRABackbone

__all__ = [
    "CNNEncoder", "CNNDecoder", "CNNDecoderResidual",
    "EndoVAEEncoder", "EndoVAEDecoder",
    "GastronetViTEncoder", "GastronetViTEncoderLoRABackbone",
]
