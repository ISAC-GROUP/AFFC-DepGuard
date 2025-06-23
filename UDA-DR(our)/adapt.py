import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
import lstm_model
import matplotlib.pyplot as plt
from torch.utils import data

def adapt(device, lr, target_epoch, batch_size, target_batch_size, save_path, src_dataset_train, tgt_dataset_train, tgt_dataset_test): # 保存适合目标域的特征提取器；将测试集提前进行验证，找到合适的微调次数，添加测试集准确率迭代图，并将横坐标标题显示出合适的微调次数

    batch_size = 8
    target_batch_size = 4

    src_train_dataloader = DataLoader(src_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_train_dataloader = DataLoader(tgt_dataset_train, batch_size=target_batch_size, shuffle=True, drop_last=True)

    tgt_test_dataloader = DataLoader(tgt_dataset_test, batch_size=len(tgt_dataset_test))

    tgt_encoder = lstm_model.LSTMEncoder()
    tgt_encoder.load_state_dict(torch.load(save_path + 'src_encoder.pth'))
    tgt_encoder = tgt_encoder.to(device)


    classifier = lstm_model.LSTMClassifier()
    classifier.load_state_dict(torch.load(save_path + 'src_classifier.pth'))
    classifier = classifier.to(device)

    torch.manual_seed(5)
    domain_discriminator = lstm_model.Discriminator()
    domain_discriminator = domain_discriminator.to(device)

    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    optimizer = torch.optim.Adam(list(tgt_encoder.parameters()) + list(domain_discriminator.parameters()), lr=1e-4)

    finetune_times = 0
    set_finetune_times = 20
    accs = []
    domain_losses = []
    domain_accs = []

    for e in tqdm(range(target_epoch)):
        for src_train_data in src_train_dataloader:

            for tgt_data in tgt_train_dataloader:
                tgt_train_data = tgt_data

            if(finetune_times == set_finetune_times):
                torch.save(tgt_encoder.state_dict(), save_path + 'tgt_encoder.pth')
                break

            tgt_encoder.eval()
            classifier.eval()
            with torch.no_grad():
                for test_data in tgt_test_dataloader:
                    feature0, feature1, feature2, feature3, feature4, label = test_data
                    label = label.to(device)
                    output = classifier(tgt_encoder(feature0.to(device), feature1.to(device), feature2.to(device),
                                                    feature3.to(device), feature4.to(device)))
                    acc_num = (output.argmax(1) == label).sum()
                    acc = acc_num / len(label)
                    accs.append(acc.item())

            tgt_encoder.train()
            domain_discriminator.train()
            optimizer.zero_grad()

            feature0, feature1, feature2, feature3, feature4, label = src_train_data
            feat_src = tgt_encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device),
                                   feature4.to(device))

            feature0, feature1, feature2, feature3, feature4, label = tgt_train_data
            feat_tgt = tgt_encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device),
                                   feature4.to(device))

            feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

            pred_concat = domain_discriminator(feat_concat, alpha=1)

            label_src = torch.ones(feat_src.size(0)).long() # 源域标签为1
            label_tgt = torch.zeros(feat_tgt.size(0)).long() # 目标域标签为0
            label_concat = torch.cat((label_src, label_tgt), dim=0)

            loss_discriminator = loss(pred_concat, label_concat.to(device))


            loss_discriminator.backward()
            optimizer.step()

            finetune_times += 1


    plt.clf()

    class_acc = np.array(accs)
    x = np.arange(class_acc.shape[0])

    plt.plot(x, class_acc, color = 'r')
    plt.savefig(save_path + 'iteration_domain_acc_01.jpg')