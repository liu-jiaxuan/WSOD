"""
ICCV 2023
Paper ID: 6886
"""


tag = 'Our_Proposed_Best_Model'

python train.py --lr_style triangle --base_lr 1e-4 --max_lr 1e-2 --epoch 40 --lr_decay_epoch 40 --tag ${tag} --color_contrast True --color_inbox True --transform_mode 3 --color_k 3 --color_d 1 --color_s 1 --color_tau 0.8 --color_temp 0.3  --contrast False --feature_down False --edge_loss True --inbox_fg True