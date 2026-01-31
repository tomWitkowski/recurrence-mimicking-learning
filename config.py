from dataclasses import dataclass
from random import randint

@dataclass
class Config:
    """
    mode
        - diff - differences historical
        -historical - previous values
    lewar - leverage
    reward_type - can be in:
        - custom - original 
        - cumulative - cumulative immediate reward
        - prod - product of percentage immediate reward
        - sharpe - sharpe ratio on immediate rewards 
        - cum_sharpe - ... 
    """
    mode: str = 'diff' # 'standard_trans'
    # mode: str = 'historical' # 'standard_trans'
    lewar: int = 1
    reverse_pair: bool = False
    # reward_type: str = 'sharpe' #  'differential_sharpe'
    # reward_type: str = 'differential_sharpe'
    # source_data: str = 'raw/stock_data/NYA.csv'
    source_data: str = 'processed/input_table_15s.csv'
    # sequential_training: str = True

    # exp_name: str = 'online_learning'
    # exp_name: str = 'test'
    # exp_name: str = 'my_method'

    ### Params RLSTM
    # exp_train_len: int = 500
    # train_batch_size: int = 500
    # exp_data_len: int =  500
    # exp_test_len: int = 3100
    # exp_return_stop: int = 0.2
    # train_max_epoch: int = 5000

    # exp_train_len: int = 1000
    # exp_data_len: int =  1000
    # train_batch_size: int =  1000
    # exp_test_len: int = 3100

    exp_train_len: int = 15_000
    exp_data_len: int =  15_000
    train_batch_size: int =  15_000
    exp_test_len: int = 3_500

    exp_return_stop: int = 0.3

    wave_CHANGE: float = 0.0005
    wave_LENGTH: int = 40

    test_during_training: bool = True

    M: int = 50

    sequential_training: str = False # True

    load_weights: bool = False

    # params experiment online learning
    # train_max_epoch: int = 150
    # reward_type: str = 'differential_sharpe' #  'differential_sharpe'
    # exp_name: str = 'online_learning' # 'my_method' # online_learning

    # # MY METHOD
    train_max_epoch: int = 5000
    reward_type: str = 'sharpe' #  'differential_sharpe'
    exp_name: str = 'my_method' # 'my_method' # online_learning




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