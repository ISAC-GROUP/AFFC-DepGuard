import random
import copy
from deal_data import MyDataSet
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import DataLoader
import lstm_model # todo
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import torch.autograd
from collections import OrderedDict
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def plot_loss_data(data, save_path):
    plt.clf()
    y = np.array(data)
    x = np.arange(y.shape[0])
    plt.plot(x, y, label = 'train_loss')
    plt.xlabel('Iteration', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.legend(loc="best")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'iteration_loss.jpg')

def plot_data(data, save_path, num):
    plt.clf()
    y_test_acc = np.array(data)
    x = np.arange(y_test_acc.shape[0])
    plt.plot(x, y_test_acc, label='test_acc')
    plt.xlabel('Iteration', fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    plt.legend(loc="best")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'iteration_accuracy_' + num + '.jpg')

def plot_confusion_matrix(y_true, y_predict, save_path):
    plt.clf()  # 清除当前的绘图
    C = confusion_matrix(y_true, y_predict)
    sns.set(font_scale=1)
    sns.heatmap(C, annot=True, cmap='Blues', fmt='.20g', annot_kws={"size": 15})  # 绘制混淆矩阵
    plt.title('Confusion Matrix')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig(save_path + 'confusion_matrix_shot16.jpg') # todo

def meta_learning(device, window, train_person, test_person, epoch, lr, save_path):

    # todo
    support_shot = 5
    query_shot = 10
    num_per_label = 16
    task_num = 30
    task_per_batch = 5
    update_step = 1
    update_lr = 1e-4

    update_step_test = 30
    test_support_shot = 30
    batch_size = 24

    task_datasets = []
    for i in range(len(train_person)):
        task_datasets.extend(utils.generate_task(train_person[i], window, task_num, num_per_label))

    task_datasets_dataloader = [DataLoader(task_dataset, batch_size=len(task_dataset)) for task_dataset in task_datasets]

    # 测试集
    test_dataset_calm_total = MyDataSet(test_person, window, set_label=0)
    test_dataset_attack_total = MyDataSet(test_person, window, set_label=1)
    torch.manual_seed(5)
    train_dataset_calm, test_dataset_calm = data.random_split(test_dataset_calm_total, [test_support_shot, len(test_dataset_calm_total) - test_support_shot])
    torch.manual_seed(5)
    train_dataset_attack, test_dataset_attack = data.random_split(test_dataset_attack_total, [test_support_shot, len(test_dataset_attack_total) - test_support_shot])
    train_calm_dataloader = DataLoader(train_dataset_calm, batch_size=batch_size, shuffle=True, drop_last=True)
    train_attack_dataloader = DataLoader(train_dataset_attack, batch_size=batch_size, shuffle=True, drop_last=True)
    test_calm_dataloader = DataLoader(test_dataset_calm, batch_size=len(test_dataset_calm))
    test_attack_dataloader = DataLoader(test_dataset_attack, batch_size=len(test_dataset_attack))

    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    test_acc = []


    model = lstm_model.Classifier()
    model.load_state_dict(torch.load(save_path + 'model.pth', map_location=device)) # todo 跑不同的实验需要换文件地址
    model = model.to(device)
    test_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for j in range(update_step_test):
        for train_calm_data, train_attack_data in zip(train_calm_dataloader, train_attack_dataloader):
            model.train()
            test_optimizer.zero_grad()
            feature0, feature1, feature2, feature3, feature4, label = utils.get_support_data(train_calm_data, train_attack_data, batch_size)
            label = label.to(device)
            output = model(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device))
            train_loss = loss(output, label)
            train_loss.backward()
            test_optimizer.step()

        model.eval()
        with torch.no_grad():
            for test_calm_data, test_attack_data in zip(test_calm_dataloader, test_attack_dataloader):
                feature0, feature1, feature2, feature3, feature4, label = utils.get_query_data(test_calm_data, test_attack_data, 0)
                label = label.to(device)
                output = model(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device))
                accuracy_sum = (output.argmax(1) == label).sum()
                accuracy = accuracy_sum / len(label)
                test_acc.append(accuracy.item())

    model.eval()
    with torch.no_grad():
        for test_calm_data, test_attack_data in zip(test_calm_dataloader, test_attack_dataloader):
            feature0, feature1, feature2, feature3, feature4, label = utils.get_query_data(test_calm_data, test_attack_data, 0)
            label = label.to(device)
            output = model(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device))
            accuracy_sum = (output.argmax(1) == label).sum()
            accuracy = accuracy_sum / len(label)

            label = label.cpu()
            output = output.argmax(1).cpu()
            precision_num = precision_score(label, output, average='macro')
            recall_num = recall_score(label, output, average='macro')
            f1_num = f1_score(label, output, average='macro')
            accuracy_score_num = accuracy_score(label, output)

    plot_data(test_acc, save_path, "15") # todo
    return precision_num, recall_num, f1_num, accuracy_score_num