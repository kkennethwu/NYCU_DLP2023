from dataloader import read_bci_data
from EEGNet import EEGNet
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, learning_rate, epochs):
    learning_rate = 1e-2
    epochs = 300

    
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossFn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs+1):
        correct_pred = 0
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            y_pred = y_pred.squeeze()
            y_pred_binary = (y_pred >= 0.5).int()
            correct_pred += (y_pred_binary == y_batch).sum().item()
            
            loss = lossFn(y_pred, y_batch)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 20 == 0:
            print("loss: ", loss.item())
            print("accuracy: ", correct_pred / 1080)

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = torch.tensor(train_data).to(device)
    train_label = torch.tensor(train_label).to(device)
    test_data = torch.tensor(test_data).to(device)
    test_label = torch.tensor(test_label).to(device) 
    loader = DataLoader(TensorDataset(train_data, train_label), batch_size=64)
    
    lr = 1e-2
    epochs = 300
    eegnet = EEGNet()
    train(eegnet, loader, lr, epochs)