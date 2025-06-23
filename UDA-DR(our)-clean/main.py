import multiprocessing
import train
import torch

def train_model(device, window, train_person, test_person, epoch, target_epoch, target_batch_size, batch_size, lr, save_path):

    acc = train.train(device, window, train_person, test_person, epoch, target_epoch, target_batch_size, batch_size, lr, save_path)
    save_acc_path = save_path + 'result4.txt'
    save_result = str(acc.item())
    save_canshu = str(epoch) + ',' + str(batch_size) + ',' + str(target_epoch) + ',' + str(target_batch_size) + ',' +str(lr) + ':'

    file = open(save_acc_path, 'a+')
    file.write(save_canshu + '\n')
    file.write(save_result + '\n')
    file.write('\n')
    file.close()

if __name__ == '__main__':

    window = '1_1'
    epoch = 13
    target_epoch = 1
    target_batch_size = 32
    batch_size = 64
    lr = 1e-4
    file_name = 'result2/' # todo result2改变模型结构

    save_path_026 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/026/'
    save_path_025 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/025/'
    save_path_024 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/024/'
    save_path_023 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/023/'
    save_path_022 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/022/'
    save_path_021 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/021/'
    save_path_020 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/020/'
    save_path_019 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/019/'
    save_path_018 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/018/'
    save_path_017 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/017/'
    save_path_015 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/015/'
    save_path_013 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/013/'
    save_path_011 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/011/'
    save_path_010 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/010/'
    save_path_009 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/009/'
    save_path_006 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/006/'
    save_path_005 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/005/'
    save_path_004 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/004/'
    save_path_003 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/003/'
    save_path_002 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/002/'

    save_path_58 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/58/'
    save_path_56 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/56/'
    save_path_52 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/52/'
    save_path_50 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/50/'
    save_path_49 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/49/'
    save_path_47 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/47/'
    save_path_45 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/45/'
    save_path_44 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/44/'
    save_path_43 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/43/'
    save_path_42 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/42/'
    save_path_40 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/40/'
    save_path_37 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/37/'
    save_path_36 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/36/'
    save_path_29 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/29/'
    save_path_28 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/28/'
    save_path_26 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/26/'
    save_path_21 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/21/'
    save_path_20 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/20/'
    save_path_17 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/17/'
    save_path_11 = '../result_exp_normal_abnormal/result_lstm_normal_abnormal_new_dann/' + window + '/11/'

    #
    touple = (
        (torch.device('cuda:0'), window,
         [['002', '003', '004', '005', '006', '009', '010', '011', '013', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['015'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_015 + file_name),

        (torch.device('cuda:0'), window,
         [['002', '003', '004', '005', '006', '009', '010', '011', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['013'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_013 + file_name),

        (torch.device('cuda:1'), window,
         [['002', '003', '004', '005', '006', '009', '010', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['011'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_011 + file_name),

        (torch.device('cuda:1'), window,
         [['002', '003', '004', '005', '006', '009', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['010'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_010 + file_name),

        (torch.device('cuda:2'), window,
         [['002', '003', '004', '005', '006', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['009'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_009 + file_name),

        (torch.device('cuda:2'), window,
         [['002', '003', '004', '005', '009', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['006'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_006 + file_name),

        (torch.device('cuda:3'), window,
         [['002', '003', '004', '006', '009', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['005'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_005 + file_name),

        (torch.device('cuda:3'), window,
         [['002', '003', '005', '006', '009', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['004'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_004 + file_name),

        (torch.device('cuda:4'), window,
         [['002', '004', '005', '006', '009', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['003'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_003 + file_name),

        (torch.device('cuda:4'), window,
         [['003', '004', '005', '006', '009', '010', '011', '013', '015', '017', '018', '019', '020', '021', '022',
           '023', '024', '025', '026'],
          ['11', '17', '20', '21', '26', '28', '29', '36', '37', '40', '42', '43', '44', '45', '47', '49', '50', '52',
           '56',
           '58']],
         [['002'], []], epoch, target_epoch, target_batch_size, batch_size, lr, save_path_002 + file_name),
    )
    for i in range(len(touple)):
        p = multiprocessing.Process(target=train_model, args=touple[i])
        p.start()
