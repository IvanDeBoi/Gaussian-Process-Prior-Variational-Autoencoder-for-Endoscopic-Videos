#------------------------------------------- GENERAL SETTINGS ----------------------------------------------------------------------

is_on_HPC = True
# is_on_HPC = False

if is_on_HPC:
    LPIPS_RES = 256 #336 takes longer and is not needed
    EPOCHS = 300
    LATENT_DIM = 256
    BATCH_SIZE = 64
    CODEBOOK_SIZE = 256
    LEARNING_RATE = 1e-3
    LORA_RANK = 8
    LORA_ALPHA = 8
    LORA_LR = 1e-5
    LORA_TARGET_KEYWORDS = ("qkv", "proj", "fc1", "fc2")
else: #quick tests
    LPIPS_RES = 32
    EPOCHS = 300
    LATENT_DIM = 256
    BATCH_SIZE = 8
    CODEBOOK_SIZE = 128
    LEARNING_RATE = 1e-3
    LORA_RANK = 6
    LORA_ALPHA = 6
    LORA_LR = 1e-5
    LORA_TARGET_KEYWORDS = ("qkv",)

# "qkv" adapts attention input projections
# "proj" adapts attention output projections
# "fc1" and "fc2" adapt the transformer MLP blocks

MUST_ALWAYS_TRAIN_FROM_SCRATCH = True
# MUST_ALWAYS_TRAIN_FROM_SCRATCH = False

# MODELS = ['VAE', 'GPVAE_HPA', 'MSVQ_VAE']
MODELS = ['VAE', 'GPVAE_HPA', 'GPVAE_SPA', 'MSVQ_VAE']
# MODELS = ['VAE', 'GPVAE_HPA']
# MODELS = ['VAE']
# MODELS = ['GPVAE_HPA']
# MODELS = ['MSVQ_VAE']

if is_on_HPC:
    FOLDER_TO_SAVE_RECONSTRUCTED = "Recon/"
    ALLOWED_PAIRS_PER_MODEL = {
        "VAE": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "GPVAE_HPA": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "GPVAE_SPA": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "MSVQ_VAE": {
            ('MSVQEnc', 'MSVQDec'),
        },
    }
else: #quick tests
    FOLDER_TO_SAVE_RECONSTRUCTED = "C:/Endo/Recon/"
    ALLOWED_PAIRS_PER_MODEL = {
        "VAE": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            # ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "GPVAE_HPA": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            # ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "GPVAE_SPA": {
            # ('CNNEnc', 'CNNDecRes'),
            ('EndoVAEEnc', 'EndoVAEDec'),
            # ('GastroEnc', 'CNNDecRes'),
            # ('GastroEnc', 'EndoVAEDec'),
            # ('GaLoRaEnc', 'CNNDecRes'),
            # ('GaLoRaEnc', 'EndoVAEDec'),
        },
        "MSVQ_VAE": {
            ('MSVQEnc', 'MSVQDec'),
        },
    }

LOSS_WEIGHTS = {
            "mse": 4.0,
            "lpips": 2.0,
            "gan": 0.001,
            "kl": 1.0,
        }


PERCENTAGE_FOR_TEST = 20
PERCENTAGE_FOR_TAIL = 0
# MUST_USE_GAP_FOR_TEST = True
MUST_USE_GAP_FOR_TEST = False
GAP_SIZE = 10

MUST_RECONSTRUCT_ALL = True
# MUST_RECONSTRUCT_ALL = False

# Save individual grid PNGs for each reconstructed frame.
# MP4 reconstruction videos are still created when this is False.
MUST_SAVE_RECONSTRUCTION_GRIDS = False

MUST_USE_LPIPS = True
# MUST_USE_LPIPS = False
if not MUST_USE_LPIPS:
    LOSS_WEIGHTS["lpips"] = 0.0

MUST_USE_MASK = True
# MUST_USE_MASK = False

# If False, raw frame numbers are used everywhere for x.
# If True, x is scaled with MinMaxScaler() to [0, 1].
USE_X_SCALING = False
# USE_X_SCALING = True

# MUST_USE_DECODER_GAN = True
MUST_USE_DECODER_GAN = False
if not MUST_USE_DECODER_GAN:
    LOSS_WEIGHTS["gan"] = 0.0

WARMUP_EPOCHS = 50
RAMP_EPOCHS = 50
GAN_WARMUP_EPOCHS = 150            # no adversarial term before this epoch
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_START_FRACTION = 0.5 # activate after this fraction of EPOCHS

#---------------------------------------------- DATA -------------------------------------------------------------------

# SUBSTRING_FILTER = 'EndoSLAM_From_Unity'
# SUBSTRING_FILTER = 'kvasir-capsule-labeled-videos_8a00709108cd4e2b_'
# SUBSTRING_FILTER = 'image_B'
# SUBSTRING_FILTER = 'image_F'
# SUBSTRING_FILTER = 'Hangar'

PREFIX = 'C:/Endo/Images from datasets/'

SUBSTRING_FILTER = 'c1_transverse2_t1_v2'
# SUBSTRING_FILTER = 'c1_cecum_t3_v3'
# SUBSTRING_FILTER = 'c1_descending_t1_v2'
# SUBSTRING_FILTER = 'c1_descending_t2_v3'
# SUBSTRING_FILTER = 'c1_rectum_t2_v2'
# SUBSTRING_FILTER = 'c1_rectum_t4_v3'
# SUBSTRING_FILTER = 'c1_transverse2_t1_v2'
# SUBSTRING_FILTER = 'c2_cecum_t3_v2'
# SUBSTRING_FILTER = 'c2_sigmoid_t2_v2'
# SUBSTRING_FILTER = 'c2_transverse2_t1_v3'


IMG_SRC_FOLDER = PREFIX + "C3VDv2/" + SUBSTRING_FILTER
# IMG_SRC_FOLDER = "Data/Blender/"
# IMG_SRC_FOLDER = "Data/EndoSLAM from unity_all gif/"
# IMG_SRC_FOLDER = "Data/kvasir-capsule-labeled-videos/labelled_videos/8a00709108cd4e2b_frames/"
# IMG_SRC_FOLDER = "Data/Pillcam Unity with black border/"
# IMG_SRC_FOLDER = "Data/Hangar/"


NEW_IMG_FOLDER = "Data/C3VDv2/c1_ascending_t1_v1" #to encode images from an unrelated video

# RESOLUTION = 320 #Endoslam
# RESOLUTION = 336 #kvasir
# RESOLUTION = 224 #Pillcam Unity
# RESOLUTION = 320 #Pillcam Unity
# RESOLUTION = 256 #Pillcam Unity
# RESOLUTION = 336 #Pillcam Unity
# RESOLUTION = 256 #Hangar
RESOLUTION = 336 #C3VDv2 (rescaled)

# SUBSTRING_NORMALISATION = "01"
# SUBSTRING_NORMALISATION = "m0s1"
SUBSTRING_NORMALISATION = "m01sG"

#------------------------------------------ FOLDERS -----------------------------------------------------------------------

# folders
FOLDER_DATA = "Data/"
FOLDER_LOGS = "Logs/"

FOLDER_OUTPUT = "Pics/"

FOLDER_POSES = "Poses/"


FOLDER_TO_SAVE_FRAMES_FOR_DEBUGGING = "Debug/"

FOLDER_TO_SAVE_MODELS = "Saved_Models/"
FOLDER_WEIGHTS = "Models/Building_Blocks/Weights"

#------------------------------------------ MISC -----------------------------------------------------------------------

MUST_CALCULATE_FID = True
# MUST_CALCULATE_FID = False

MUST_CALCULATE_SSIM = True
# MUST_CALCULATE_SSIM = False

MUST_CALCULATE_PSNR = True
# MUST_CALCULATE_PSNR = False

# MUST_PLOT_COVARIANCE = True
MUST_PLOT_COVARIANCE = False

# MUST_PLOT_2D_LATENT = True
MUST_PLOT_2D_LATENT = False

MUST_PLOT_PCA = False
MUST_TIME_ENCODE_NEW_IMAGES = False

# MODELS = ['PCA', 'CVAE', 'TSNE', 'UMAP', 'GPLVM', 'GPRDM', 'GPDM', 'DGPLVM', 'GPPVAE'] # none of these work well
# MODELS = ['VAE', 'GPVAE_Casale', 'SVGPVAE_Jazbec', 'GPVAE_SPA', 'GPVAE_HPA']
# MODELS = ['VAE', 'SVGPVAE_Jazbec', 'GPVAE_SPA', 'GPVAE_HPA']
# MODELS = ['SVGPVAE_Jazbec', 'GPVAE_SPA', 'GPVAE_HPA']
# MODELS = ['VAE', 'GPVAE_SPA', 'GPVAE_HPA']

ENCODERS = ['CNNEnc', 'EndoVAEEnc', 'GastroEnc', 'GaLoRaEnc', 'MSVQEnc']

DECODERS = ['CNNDecRes', 'EndoVAEDec', 'MSVQDec']

# image / data options
MUST_PRESERVE_COLOR = True

KERNEL = 'CAUCHY'
# KERNEL = 'MATERN'
# KERNEL = 'CAUCHY'

NR_OF_NEIGHBOURS = 50 # neighbours in HPA and SPA

import torch
import numpy as np

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PCA_TARGET_VARIANCE = 0.95 #CB
# PCA_TARGET_VARIANCE = 50

# MIN_SPATIAL = 8
MIN_SPATIAL = 8

GASTRONET_MEAN = [0.64041256, 0.36125767, 0.31330117]
GASTRONET_STD = [0.18983584, 0.15554344, 0.14093774]

# randomness
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)

# ------------------------------------------ ADVERSARIAL / GAN -----------------------------------------------------------------------

GAN_TYPE = "patchgan"            # currently only patchgan implemented
GAN_LOSS = "hinge"               # options: "hinge", "bce"
#GAN_WEIGHT = 0.01                # weight added to decoder/generator loss
GAN_DISC_WEIGHT = 1.0            # discriminator loss scaling
GAN_LR = 2e-4
GAN_BETAS = (0.5, 0.999)

GAN_BASE_CHANNELS = 32
GAN_NUM_LAYERS = 3               # 4-5 is typical for 256-336-ish images
GAN_NORM = "instance"            # "instance", "batch", "group", "none"

GAN_USE_MASK = True              # zero background / invalid pixels before D
GAN_DETACH_RECON_FOR_D = True    # standard GAN training
GAN_LABEL_SMOOTHING = 0.0        # only used for BCE variant
GAN_DISC_UPDATES = 1             # D steps per generator step

GAN_FEATURE_MATCHING_WEIGHT = 0.1  # optional, keep 0 initially










