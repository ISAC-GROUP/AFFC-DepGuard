from torch.utils.data import DataLoader
import torch
import lstm_model

def test(device, save_path, tgt_dataset_test):
    tgt_test_dataloader = DataLoader(tgt_dataset_test, batch_size=len(tgt_dataset_test))

    encoder = lstm_model.LSTMEncoder()
    encoder.load_state_dict(torch.load(save_path + 'tgt_encoder.pth'))
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
            output = classifier(encoder(feature0.to(device), feature1.to(device), feature2.to(device), feature3.to(device), feature4.to(device)))
            accuracy_num = (output.argmax(1) == label).sum()
            accuracy = accuracy_num / len(label)

    return accuracy