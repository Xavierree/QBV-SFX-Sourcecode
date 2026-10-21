BEATs (Chen et al., 2022; https://github.com/microsoft/unilm/tree/master/beats), MIT License (see LICENSE).
Only the encoder files the D1 backend imports are kept, unchanged: BEATs.py, backbone.py, modules.py. They import each other by bare name, so
sfxdeploy/d1_towers.py puts this folder on sys.path. The pretrained BEATs checkpoint is not needed: the fine-tuned towers in models/d1/ hold every weight.
