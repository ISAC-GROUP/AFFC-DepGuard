import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, person, window):
        data0, data1, data2, data3, data4, label_sum, label_emotion_sum = [], [], [], [], [], [], []
        for i in range(len(person[0])): # 处理正常人数据
            feature_path = '../dataset_neutral_window/' + person[0][i] + '/' + window +  '/feature.npy'
            data = np.load(feature_path, allow_pickle=True)

            data_length = len(data[0])
            for j in range(data_length):
                data0.append(self.norm(data[0][j]))
                data1.append(self.norm(data[1][j]))
                data2.append(self.norm(data[2][j]))
                data3.append(self.norm(data[3][j]))
                data4.append(self.norm(data[4][j]))
                label_sum.append(0)



        for i in range(len(person[1])):
            feature_path = '../dataset_depression_window/' + person[1][i] + '/' + window  + '/feature.npy'
            label_path = '../dataset_depression_window/' + person[1][i] + '/' + window + '/label.npy'

            data = np.load(feature_path, allow_pickle=True)
            label = np.load(label_path)

            data_length = len(data[0])
            attack_sum = 500
            calm_sum = 500
            for j in range(data_length):
                if(label[j] == 0):
                    data0.append(self.norm(data[0][j]))
                    data1.append(self.norm(data[1][j]))
                    data2.append(self.norm(data[2][j]))
                    data3.append(self.norm(data[3][j]))
                    data4.append(self.norm(data[4][j]))
                    label_sum.append(1)
                    calm_sum = calm_sum - 1
                    if(calm_sum <= 0):
                         break
            for j in range(data_length):
                if(label[j] == 1):
                    data0.append(self.norm(data[0][j]))
                    data1.append(self.norm(data[1][j]))
                    data2.append(self.norm(data[2][j]))
                    data3.append(self.norm(data[3][j]))
                    data4.append(self.norm(data[4][j]))
                    label_sum.append(1)
                    attack_sum = attack_sum - 1
                    if(attack_sum <= 0):
                         break
        self.data0 = torch.tensor(data0, dtype=torch.float32)
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.data3 = torch.tensor(data3, dtype=torch.float32)
        self.data4 = torch.tensor(data4, dtype=torch.float32)
        self.label = torch.tensor(label_sum, dtype=torch.long)

    def __getitem__(self, item):
        return self.data0[item], self.data1[item], self.data2[item], self.data3[item], self.data4[item], self.label[item]

    def __len__(self):
        return self.data0.shape[0]

    def norm(self, x):
        return (2 * (x - np.min(x))) / (np.max(x) - np.min(x)) - 1