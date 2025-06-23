from deal_data import MyDataSet
from torch.utils import data
import torch
import adapt
import test


def train(device, window, train_person, test_person, epoch, target_epoch, target_batch_size, batch_size, lr, save_path):

    src_dataset = MyDataSet(train_person, window)
    tgt_dataset = MyDataSet(test_person, window)

    src_dataset_train_size = int(len(src_dataset) * 0.8)
    src_dataset_valid_size = len(src_dataset) - src_dataset_train_size
    torch.manual_seed(5)
    src_dataset_train, src_dataset_valid = data.random_split(src_dataset, [src_dataset_train_size, src_dataset_valid_size])

    tgt_dataset_train_size = 5
    tgt_dataset_test_size = len(tgt_dataset) - tgt_dataset_train_size
    torch.manual_seed(5)
    tgt_dataset_train, tgt_dataset_test = data.random_split(tgt_dataset, [tgt_dataset_train_size, tgt_dataset_test_size])


    adapt.adapt(device, lr, target_epoch, batch_size, target_batch_size, save_path, src_dataset_train, tgt_dataset_train, tgt_dataset_test)

    acc = test.test(device, save_path, tgt_dataset_test)

    return acc