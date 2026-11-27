# Vendored DINOv2

This folder is a vendored copy of parts of the DINOv2 code from https://github.com/facebookresearch/dinov2
(`layers/`, `models/`, `utils/`), kept unchanged so that its relative imports keep working. It is licensed under the
Apache License, Version 2.0: the source files carry the header "Copyright (c) Meta Platforms, Inc. and affiliates. This
source code is licensed under the Apache License, Version 2.0 found in the LICENSE file in the root directory of this
source tree" (the upstream LICENSE file itself is not included in this copy). It is used only by the GastroEnc / GaLoRaEnc
encoders (`gpvae_endo/networks/vit.py`).
