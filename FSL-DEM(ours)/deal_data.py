import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, person, window, set_label):
        data0, data1, data2, data3, data4, label_sum = [], [], [], [], [], []

        for i in range(len(person)):

            feature_path = '../dataset_attack/' + person[i] + '/feature.npy'
            label_path = '../dataset_attack/' + person[i] + '/label.npy'

            data = np.load(feature_path, allow_pickle=True)
            label = np.load(label_path)

            data_length = len(label)
            if(set_label == 10):
                for j in range(data_length): # todo
                    data0.append(data[0][j])
                    data1.append(data[1][j])
                    data2.append(data[2][j])
                    data3.append(data[3][j])
                    data4.append(data[4][j])
                    label_sum.append(label[j])
            else:
                for j in range(data_length): # todo
                    if (label[j] == set_label):
                        data0.append(data[0][j])
                        data1.append(data[1][j])
                        data2.append(data[2][j])
                        data3.append(data[3][j])
                        data4.append(data[4][j])
                        label_sum.append(label[j])

        self.data0 = torch.tensor(data0, dtype=torch.float32)
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.data3 = torch.tensor(data3, dtype=torch.float32)
        self.data4 = torch.tensor(data4, dtype=torch.float32)
        self.label = torch.tensor(label_sum, dtype=torch.long)

    def __getitem__(self, item):
        return self.data0[item], self.data1[item], self.data2[item], self.data3[item], self.data4[item], self.label[item]

    def __len__(self):
        return self.data0.shape[0] # todo

    def norm(self, x):
        return (2 * (x - np.min(x))) / (np.max(x) - np.min(x)) - 1