from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import lstm_model
import test

def pre_train_after_test(device, save_path, tgt_dataset):

    tgt_test_dataloader = DataLoader(tgt_dataset, batch_size=len(tgt_dataset))

    encoder = lstm_model.LSTMEncoder()
    encoder.load_state_dict(torch.load(save_path + 'src_encoder.pth'))
    encoder = encoder.to(device)
    classifier = lstm_model.LSTMClassifier()
    classifier.load_state_dict(torch.load(save_path + 'src_classifier.pth'))
    classifier = classifier.to(device)

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for test_data in tgt_test_dataloader:
            feature0, feature1, feature2, feature3, feature4, label = test_data
            label = label.to(device)
            output = classifier(
                encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device),
                        feature4.to(device)))
            accuracy_num = (output.argmax(1) == label).sum()
            accuracy = accuracy_num / len(label)

    return accuracy

def pre_train(device, lr, epoch, batch_size, save_path, src_dataset_train, src_dataset_valid, tgt_dataset):
    src_train_dataloader = DataLoader(src_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    src_valid_dataloader = DataLoader(src_dataset_valid, batch_size=len(src_dataset_valid))

    torch.manual_seed(5)
    src_encoder = lstm_model.LSTMEncoder()
    src_encoder = src_encoder.to(device)
    torch.manual_seed(5)
    src_classifier = lstm_model.LSTMClassifier()
    src_classifier = src_classifier.to(device)
    # 损失函数与优化器
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    optimizer = torch.optim.Adam(list(src_encoder.parameters()) + list(src_classifier.parameters()), lr=lr)
    # 开始模型训练
    iteration_num = 1  # 定义迭代次数
    min_loss = 100000  # 初始化定义损失值
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    for e in tqdm(range(epoch)):
        for train_data in src_train_dataloader:
            print("----------Starting {} Training----------".format(iteration_num))
            src_encoder.train()
            src_classifier.train()
            optimizer.zero_grad()
            feature0, feature1, feature2, feature3, feature4, label = train_data
            label = label.to(device)
            output = src_classifier(src_encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device)))
            train_loss = loss(output, label)
            accuracy_num = (output.argmax(1) == label).sum()
            train_accuracy = accuracy_num / len(label)

            train_loss.backward()
            optimizer.step()
            print("For {} training，train_loss：{}".format(iteration_num, train_loss.item()))
            # 验证开始
            print("----------Starting {} validation ----------".format(iteration_num))
            src_encoder.eval()
            src_classifier.eval()
            with torch.no_grad():
                for valid_data in src_valid_dataloader:
                    feature0, feature1, feature2, feature3, feature4, label = valid_data
                    label = label.to(device)
                    output = src_classifier(src_encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device)))
                    accuracy_num = (output.argmax(1) == label).sum()
                    valid_loss = loss(output, label)
                    accuracy = accuracy_num / len(label)
                    print("For {} iteration, valid_loss：{},valid_accuracy：{}".format(iteration_num, valid_loss.item(), accuracy))
            if (valid_loss.item() < min_loss):
                min_loss = valid_loss.item()
                min_loss_epoch = e
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(src_encoder.state_dict(), save_path + 'src_encoder.pth')
                torch.save(src_classifier.state_dict(), save_path + 'src_classifier.pth')
            iteration_num += 1
            train_losses.append(train_loss.item())
            valid_losses.append(valid_loss.item())
            train_acc.append(train_accuracy.item())
            valid_acc.append(accuracy.item())

    test_acc = pre_train_after_test(device, save_path, tgt_dataset)

    y_train_loss = np.array(train_losses)
    y_valid_loss = np.array(valid_losses)
    x = np.arange(y_train_loss.shape[0])
    plt.plot(x, y_train_loss, label='train_loss')
    plt.plot(x, y_valid_loss, label='valid_loss')
    plt.xlabel('Iteration', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.legend(loc="best")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'iteration_loss.jpg')

    plt.clf()
    y_train_acc = np.array(train_acc)
    y_valid_acc = np.array(valid_acc)
    x = np.arange(y_train_acc.shape[0])
    plt.plot(x, y_train_acc, label='train_acc')
    plt.plot(x, y_valid_acc, label='valid_acc')
    plt.xlabel('Iteration', fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    plt.title(str(test_acc.item())) #
    plt.legend(loc="best")
    plt.savefig(save_path + 'iteration_accuracy.jpg')
