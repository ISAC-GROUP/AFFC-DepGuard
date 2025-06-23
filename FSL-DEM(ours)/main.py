import multiprocessing
import meta_learning
import torch

def train_model(device, window, train_person, test_person, epoch, lr, save_path):
    # meta_learning.meta_learning(device, window, train_person, test_person, epoch, lr, save_path)

    precision_num, recall_num, f1_num, accuracy_score_num = meta_learning.meta_learning(device, window, train_person, test_person, epoch, lr, save_path)

    save_acc_path = save_path + 'result15.txt'
    file = open(save_acc_path, 'a+')
    canshu = "shot(per class) set 30，微调30次"

    file.write(canshu + '\n')
    file.write("precision_score:" + str(precision_num) + '\n')
    file.write("recall_score:" + str(recall_num) + '\n')
    file.write("f1_score:" + str(f1_num) + '\n')
    file.write("accuracy_score:" + str(accuracy_score_num) + '\n')
    file.write('\n')
    file.close()


if __name__ == '__main__':
    window = '1_1'

    epoch = 8 # todo 40 25 17 12 9
    lr = 1e-4 #
    filename = 'result1/'

    save_path_58 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/58/'
    save_path_56 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/56/'
    save_path_50 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/50/'
    save_path_49 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/49/'
    save_path_47 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/47/'
    save_path_45 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/45/'
    save_path_44 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/44/'
    save_path_43 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/43/'
    save_path_42 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/42/'
    save_path_40 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/40/'
    save_path_37 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/37/'
    save_path_36 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/36/'
    save_path_29 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/29/'
    save_path_28 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/28/'
    save_path_21 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/21/'
    save_path_20 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/20/'
    save_path_17 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/17/'
    save_path_11 = '../result_exp_calm_attack/results_lstm_calm_attack_maml++/' + window + '/11/'


    touple_beforeData = (
        (torch.device('cuda:0'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
        '42', '43', '44', '45', '47', '49', '50', '56'], ['58'], epoch, lr, save_path_58 + filename),

        (torch.device('cuda:0'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '44', '45', '47', '49', '50', '58'], ['56'], epoch, lr,
         save_path_56 + filename),

        (torch.device('cuda:0'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '44', '45', '47', '49', '56', '58'], ['50'], epoch, lr,
         save_path_50 + filename),

        (torch.device('cuda:0'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '44', '45', '47', '50', '56', '58'], ['49'], epoch, lr,
         save_path_49 + filename),

        (torch.device('cuda:1'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '44', '45', '49', '50', '56', '58'], ['47'], epoch, lr,
         save_path_47 + filename),

        (torch.device('cuda:1'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '44', '47', '49', '50', '56', '58'], ['45'], epoch, lr,
         save_path_45 + filename),

        (torch.device('cuda:1'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '43', '45', '47', '49', '50', '56', '58'], ['44'], epoch, lr,
         save_path_44 + filename),

        (torch.device('cuda:1'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '42', '44', '45', '47', '49', '50', '56', '58'], ['43'], epoch, lr,
         save_path_43 + filename),

        (torch.device('cuda:2'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '40',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['42'], epoch, lr,
         save_path_42 + filename),

        (torch.device('cuda:2'), window, ['11', '17', '20', '21', '28', '29', '36', '37', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['40'], epoch, lr,
         save_path_40 + filename),

        (torch.device('cuda:2'), window, ['11', '17', '20', '21', '28', '29', '36', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['37'], epoch, lr,
         save_path_37 + filename),

        (torch.device('cuda:2'), window, ['11', '17', '20', '21', '28', '29', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['36'], epoch, lr,
         save_path_36 + filename),

        (torch.device('cuda:3'), window, ['11', '17', '20', '21', '28', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['29'], epoch, lr,
         save_path_29 + filename),

        (torch.device('cuda:3'), window, ['11', '17', '20', '21', '29', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['28'], epoch, lr,
         save_path_28 + filename),

        (torch.device('cuda:3'), window, ['11', '17', '20', '28', '29', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['21'], epoch, lr,
         save_path_21 + filename),

        (torch.device('cuda:3'), window, ['11', '17', '21', '28', '29', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['20'], epoch, lr,
         save_path_20 + filename),

        (torch.device('cuda:4'), window, ['11', '20', '21', '28', '29', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['17'], epoch, lr,
         save_path_17 + filename),

        (torch.device('cuda:4'), window, ['17', '20', '21', '28', '29', '36', '37', '40', '42',
                                          '43', '44', '45', '47', '49', '50', '56', '58'], ['11'], epoch, lr,
         save_path_11 + filename),

    )

    for i in range(len(touple_beforeData)): #
        p = multiprocessing.Process(target=train_model, args=touple_beforeData[i])
        p.start()