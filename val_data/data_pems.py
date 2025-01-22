import os
import torch
import pickle
import argparse
import numpy as np
from datetime import datetime, timedelta

def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int, norm_each_channel: int = False) -> np.array:
    """
    Standardized data.

    parameter:
        data (np.array): Raw time series data.
        output_dir (str): Output directory path.
        train_index (list): An index of the training data.
        history_seq_len (int): Historical sequence length.
        future_seq_len (int): Future sequence length.
        norm_each_channel (bool): Whether to perform separate normalization for each channel.

    return:
        np.array: Normalized time series data.
    """
    if_rescale = not norm_each_channel  # Whether to evaluate on rescaled data

    # Getting training data
    data_train = data[:train_index[-1][1], ...]

    mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("The mean of the training data:", mean)
    print("Standard deviation of the training data:", std)

    # Save the normalization parameters
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    
    # Save the normalization parameters to a file
    with open(output_dir + "/scaler_in_{0}_out_{1}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(scaler, f)

    # Define the normalization function
    def normalize(x):
        return (x - mean) / std

    # Normalize the data
    data_norm = normalize(data)
    return data_norm


def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    mean, std = kwargs["mean"], kwargs["std"]
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).unsqueeze(0)
        std = torch.from_numpy(std).type_as(data).to(data.device).unsqueeze(0)
    data = data * std
    data = data + mean
    return data


def generate_data(args: argparse.Namespace):
    """
    Preprocess and generate train/validation/test datasets.

    parameter:
        args (argparse.Namespace): Configuration parameters for preprocessing
    """

    # Extract parameter configuration
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel

    # read data
    data = np.load(data_file_path)['data'][...,0] # 2维npz / 3维npz
    # data = np.transpose(data, (1, 0))
    data = np.expand_dims(data, axis=-1) 
    print("Raw time series shape: {0}".format(data.shape))

    # Splitting the dataset
    l, n, f= data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("Number of training samples: {0}".format(train_num))
    print("Number of validation samples: {0}".format(valid_num))
    print("Number of test samples: {0}".format(test_num))

    # 生成索引列表
    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    print(len(train_index))
    # 归一化数据
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)
    feature_list = [data_norm]
    print("data_shape",data_norm.shape)

    # 添加时间特征
    start_date = datetime(2016, 7, 1)  # PEMS08
    # start_date = datetime(2016, 4, 1)  # bike_drop、bike_pick、taxi_drop、taxi_pick
    # start_date = datetime(2017, 5, 1)  # PEMS07M
    # start_date = datetime(2020, 3, 1)  # CAD3
    # start_date = datetime(2021, 1, 1)  # CHI_TAXI
    time_features = []
    for i in range(data_norm.shape[0]):
        current_date = start_date + timedelta(minutes=i * 1440 / args.steps_per_day)  # 根据采样频率5分钟来计算每个时间步对应的时间
        month_feature = current_date.month / 12 - 0.5
        day_feature = current_date.day / 31 - 0.5
        weekday_feature = current_date.weekday() / 6 - 0.5
        hour_feature = current_date.hour / 23 - 0.5
        minute_feature = current_date.minute / 59 - 0.5
        second_feature = current_date.second / 59 - 0.5
        time_features.append([month_feature, day_feature, weekday_feature, hour_feature, minute_feature, second_feature])
    time_features = np.array(time_features)[:, np.newaxis, :]
    feature_list.append(time_features.repeat(data_norm.shape[1], axis=1))

    processed_data = np.concatenate(feature_list, axis=-1)
    print("processed_data_shape",processed_data.shape)
    
    # 保存数据和索引
    index = {"train": train_index, "valid": valid_index, "test": test_index}
    with open(f"{output_dir}/index_in_{history_seq_len}_out_{future_seq_len}.pkl", "wb") as f:
        pickle.dump(index, f)

    data = {"processed_data": processed_data}
    with open(f"{output_dir}/data_in_{history_seq_len}_out_{future_seq_len}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # 窗口大小用于生成历史序列和目标序列
    data_list = ['CAir_PM']

    for data in data_list:
        HISTORY_SEQ_LEN = 12
        FUTURE_SEQ_LEN = 12

        TRAIN_RATIO = 0.6
        VALID_RATIO = 0.2
        TARGET_CHANNEL = [0]
        # STEPS_PER_DAY = 288 # PEMS08 PEMS04 PEMS07M CAD3 CAD5
        # STEPS_PER_DAY = 48 # CHI_TAXI bike_drop bike_pick taxi_drop taxi_pick CAir_AQI CAir_PM
        STEPS_PER_DAY = 24 # CAir_AQI CAir_PM

        DATASET_NAME = data
        TOD = True
        DOW = True

        OUTPUT_DIR = f"{DATASET_NAME}"
        DATA_FILE_PATH = f"{DATASET_NAME}.npz"

        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录。")
        parser.add_argument("--data_file_path", type=str, default=DATA_FILE_PATH, help="原始交通数据路径。")
        parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN, help="序列长度。")
        parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN, help="序列长度。")
        parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY, help="每日步数。")
        parser.add_argument("--tod", type=bool, default=TOD, help="添加时间特征。")
        parser.add_argument("--dow", type=bool, default=DOW, help="添加星期特征。")
        parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL, help="选定的通道。")
        parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="训练比例")
        parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO, help="验证比例。")
        parser.add_argument("--norm_each_channel", type=float, help="归一化每个通道。")
        args = parser.parse_args()

        # 打印参数
        print("-" * (20 + 45 + 5))
        for key, value in sorted(vars(args).items()):
            print("|{0:>20} = {1:<45}|".format(key, str(value)))
        print("-" * (20 + 45 + 5))

        # 创建输出目录
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.norm_each_channel = False
        generate_data(args)
