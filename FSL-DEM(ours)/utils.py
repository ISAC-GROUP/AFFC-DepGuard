from torch.utils import data
import torch
import torch.nn.functional as F
from deal_data import MyDataSet
import random


def computing_proto(support_output, support_label):

    num_classes = torch.unique(support_label).size(0)
    proto = torch.zeros(num_classes, support_output.size(1))
    for class_index in range(num_classes):
        class_samples_index = (support_label == class_index).nonzero(as_tuple=True)[0]
        class_samples = support_output[class_samples_index]
        class_proto = torch.mean(class_samples,dim=0)
        proto[class_index] = class_proto
    return proto

def computing_distance(query_output, proto):
    distance = -torch.norm(query_output.unsqueeze(1) - proto, p=2, dim=2)
    output = F.softmax(distance, dim=1)
    return output

def get_support_data(data_calm, data_attack, shot):

    feature0_calm, feature1_calm, feature2_calm, feature3_calm, feature4_calm, label_calm = data_calm
    feature0_attack, feature1_attack, feature2_attack, feature3_attack, feature4_attack, label_attack = data_attack
    feature0 = torch.cat((feature0_calm[:shot], feature0_attack[:shot]), dim=0)
    feature1 = torch.cat((feature1_calm[:shot], feature1_attack[:shot]), dim=0)
    feature2 = torch.cat((feature2_calm[:shot], feature2_attack[:shot]), dim=0)
    feature3 = torch.cat((feature3_calm[:shot], feature3_attack[:shot]), dim=0)
    feature4 = torch.cat((feature4_calm[:shot], feature4_attack[:shot]), dim=0)
    label = torch.cat((label_calm[:shot], label_attack[:shot]), dim=0)

    return feature0, feature1, feature2, feature3, feature4, label

def get_query_data(data_calm, data_attack, shot):

    feature0_calm, feature1_calm, feature2_calm, feature3_calm, feature4_calm, label_calm = data_calm
    feature0_attack, feature1_attack, feature2_attack, feature3_attack, feature4_attack, label_attack = data_attack
    feature0 = torch.cat((feature0_calm[shot:], feature0_attack[shot:]), dim=0)
    feature1 = torch.cat((feature1_calm[shot:], feature1_attack[shot:]), dim=0)
    feature2 = torch.cat((feature2_calm[shot:], feature2_attack[shot:]), dim=0)
    feature3 = torch.cat((feature3_calm[shot:], feature3_attack[shot:]), dim=0)
    feature4 = torch.cat((feature4_calm[shot:], feature4_attack[shot:]), dim=0)
    label = torch.cat((label_calm[shot:], label_attack[shot:]), dim=0)

    return feature0, feature1, feature2, feature3, feature4, label

def get_data(data, start_index_calm, end_index_calm, start_index_attack, end_index_attack):

    feature0, feature1, feature2, feature3, feature4, label = data

    feature0 = torch.cat((feature0[start_index_calm:end_index_calm], feature0[start_index_attack:end_index_attack]), dim=0)
    feature1 = torch.cat((feature1[start_index_calm:end_index_calm], feature1[start_index_attack:end_index_attack]), dim=0)
    feature2 = torch.cat((feature2[start_index_calm:end_index_calm], feature2[start_index_attack:end_index_attack]), dim=0)
    feature3 = torch.cat((feature3[start_index_calm:end_index_calm], feature3[start_index_attack:end_index_attack]), dim=0)
    feature4 = torch.cat((feature4[start_index_calm:end_index_calm], feature4[start_index_attack:end_index_attack]), dim=0)
    label = torch.cat((label[start_index_calm:end_index_calm], label[start_index_attack:end_index_attack]), dim=0)

    return feature0, feature1, feature2, feature3, feature4, label

def generate_task(task_person, window, task_num, num_per_label):

    person = [task_person]
    src_dataset_calm = MyDataSet(person, window, set_label=0)
    src_dataset_attack = MyDataSet(person, window, set_label=1)
    task_datasets = []

    for i in range(task_num):
        random_seed = random.randint(0, 10000)
        torch.manual_seed(random_seed)
        task_sample_calm, src_dataset_calm_part = data.random_split(src_dataset_calm, [num_per_label, len(src_dataset_calm)-num_per_label])
        torch.manual_seed(random_seed)
        task_sample_attack, src_dataset_attack_part = data.random_split(src_dataset_attack, [num_per_label, len(src_dataset_attack)-num_per_label])
        task_sample = data.ConcatDataset([task_sample_calm, task_sample_attack])
        task_datasets.append(task_sample)

    return task_datasets

def generate_task_random(task_person, window, task_num, num_per_label): # 按照随机来生成任务

    src_dataset_calm = MyDataSet(task_person, window, set_label=0)
    src_dataset_attack = MyDataSet(task_person, window, set_label=1)
    task_datasets = []

    for i in range(task_num):
        random_seed = random.randint(0, 10000)
        torch.manual_seed(random_seed)
        task_sample_calm, src_dataset_calm_part = data.random_split(src_dataset_calm, [num_per_label, len(src_dataset_calm)-num_per_label])
        torch.manual_seed(random_seed)
        task_sample_attack, src_dataset_attack_part = data.random_split(src_dataset_attack, [num_per_label, len(src_dataset_attack)-num_per_label])
        task_sample = data.ConcatDataset([task_sample_calm, task_sample_attack])
        task_datasets.append(task_sample)

    return task_datasets

