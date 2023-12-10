from dataclasses import dataclass
from random import randint

@dataclass
class Config:
    """
    lewar - leverage
    """
    lewar: int = 1
    reverse_pair: bool = False
    exp_name: str = 'Smith'
    exp_train_len: int = 5000
    exp_test_len: int = 500
    exp_data_len: int = 18_500
    exp_return_stop: int = 150
    wave_CHANGE: float = 0.004
    wave_LENGTH: int = 20
    train_batch_size: int = 200 # 5000
    train_max_epoch: int = 10000


# @dataclass
# class Config:
#     lewar: int = 1
#     reverse_pair: bool = False
#     exp_name: str = 'Smith'
#     exp_train_len: int = 250_000
#     exp_test_len: int = 50_000
#     exp_data_len: int = 1_000_000
#     exp_return_stop: int = 150
#     wave_CHANGE: float = 0.005
#     wave_LENGTH: int = 20
#     train_batch_size: int = 5000 # 5000
#     train_max_epoch: int = 10000