from dataclasses import dataclass
from random import randint

@dataclass
class Config:
    """
    mode
        - diff - differences historical
        - historical - previous values
    lewar - leverage
    reward_type - can be in:
        - custom - original 
        - cumulative - cumulative immediate reward
        - prod - product of percentage immediate reward
        - sharpe - sharpe ratio on immediate rewards 
        - cum_sharpe - ... 
    M - number of features
    """
    gpu: bool = False

    mode: str = 'diff'
    lewar: int = 1
    reverse_pair: bool = False
    
    source_data: str = 'processed/input_table_15s.csv'

    exp_train_len: int = 15_000
    exp_data_len: int =  15_000
    train_batch_size: int =  15_000
    exp_test_len: int = 3_500
    M: int = 50

    n_experiments: int = 50
    exp_return_stop: int = 0.3

    # used for data processing if mode is 'standard_trans'
    wave_CHANGE: float = 0.0005
    wave_LENGTH: int = 40

    test_during_training: bool = True
    offline_training: str = False # True
    load_weights: bool = False

    # params experiment online learning
    # train_max_epoch: int = 150
    # reward_type: str = 'differential_sharpe' #  'differential_sharpe'
    # exp_name: str = 'online_learning' # 'my_method' # online_learning

    # # MY METHOD
    train_max_epoch: int = 5000
    reward_type: str = 'sharpe' #  'differential_sharpe'
    exp_name: str = 'my_method' # 'my_method' # online_learning



