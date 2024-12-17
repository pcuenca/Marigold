# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .cocogold_trainer import CocogoldTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "CocogoldTrainer": CocogoldTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
