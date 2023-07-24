from dataloader import read_bci_data
from EEGNet import EEGNet
from DeepConvNet import DeepConvNet, ActivationLayer
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, test_loader, learning_rate, epochs, activation):
    
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True, eps=1e-4)
    # lossFn = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(1, epochs+1):
        correct_pred = 0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            
            y_pred_binary = torch.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)
            correct_pred += (y_pred_binary == y_batch).sum().item()
        train_acc = correct_pred / 1080
        train_acc_list.append(train_acc)
        
        epoch_list.append(epoch)
        val_loss, test_acc = test(model, test_loader)  
        test_acc_list.append(test_acc)
        
        model.train()
        scheduler.step(val_loss)
         
        if epoch % 20 == 0:
            print(f"##### epoch: {epoch} | {epochs} #####")
            print(f"train_loss: {loss.item()}, train_accuracy: {correct_pred / 1080}")
            print("test accuracy: ", test_acc)    
    
    plt.plot(epoch_list, train_acc_list, label=f"{activation}_train")    
    plt.plot(epoch_list, test_acc_list, label=f"{activation}_test")
    
        
         
            
def test(model, loader):
    with torch.no_grad():
        model.eval()
        correct_pred = 0
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            y_pred_binary = torch.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)
            correct_pred += (y_pred_binary == y_batch).sum().item()
            loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
            # scheduler.step(loss)
        return loss, correct_pred / 1080
        # print("accuracy: ", correct_pred / 1080)
            
    

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = torch.tensor(train_data).to(device)
    train_label = torch.tensor(train_label, dtype=torch.int64).to(device)
    test_data = torch.tensor(test_data).to(device)
    test_label = torch.tensor(test_label, dtype=torch.int64).to(device) 
    train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=1080)
    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=1080)
    lr = 1e-2
    epochs = 1800
    
    print("########## Running ##########")
    eegnet = EEGNet(torch.nn.ReLU())
    train(eegnet, train_loader, test_loader, lr, epochs, "EGG_ReLU")
    torch.save(eegnet.state_dict(), "./weights/EEG_ReLU")
    print("########## Finished ##########")
    print("########## Running ##########")
    eegnet = EEGNet(torch.nn.ELU())
    train(eegnet, train_loader, test_loader, lr, epochs, "EGG_ELU")
    torch.save(eegnet.state_dict(), "./weights/EEG_ELU")
    print("########## Finished ##########")
    print("########## Running ##########")
    eegnet = EEGNet(torch.nn.LeakyReLU())
    train(eegnet, train_loader, test_loader, lr, epochs, "EGG_LeakyReLU")
    torch.save(eegnet.state_dict(), "./weights/EEG_LeakyReLU")
    print("########## Finished ##########")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Activation Function Comparison(EEGNet)')
    plt.legend()
    plt.show()  
    
    print("########## Running ##########")
    deepConvNet1 = DeepConvNet(torch.nn.ReLU())
    train(deepConvNet1, train_loader, test_loader, lr, epochs, "deep_ReLU")
    torch.save(deepConvNet1.state_dict(), "./weights/deep_ReLU")
    print("########## Finished ##########")
    print("########## Running ##########")
    deepConvNet1 = DeepConvNet(torch.nn.ELU(alpha=1.0))
    train(deepConvNet1, train_loader, test_loader, lr, epochs, "deep_ELU")
    torch.save(deepConvNet1.state_dict(), "./weights/deep_ELU")
    print("########## Finished ##########")
    print("########## Running ##########")
    deepConvNet1 = DeepConvNet(torch.nn.LeakyReLU())
    train(deepConvNet1, train_loader, test_loader, lr, epochs, "deep_LeakyReLU")
    torch.save(deepConvNet1.state_dict(), "./weights/deep_LeakyReLU")
    print("########## Finished ##########")
    
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Activation Function Comparison(DeepConvNet)')
    plt.legend()
    plt.show()  
    
    # model = EEGNet(torch.nn.ReLU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/EEG_ReLU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("EEG_ReLU",test_acc)
    
    # model = EEGNet(torch.nn.LeakyReLU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/EEG_LeakyReLU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("EEG_LeakyReLU",test_acc)
    
    # model = EEGNet(torch.nn.ELU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/EEG_ELU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("EEG_ELU",test_acc)
    
    # model = DeepConvNet(torch.nn.ReLU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/deep_ReLU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("deep_ReLU",test_acc)
    
    # model = DeepConvNet(torch.nn.LeakyReLU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/deep_LeakyReLU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("deep_LeakyReLU",test_acc)
    
    # model = DeepConvNet(torch.nn.ELU())
    # model = model.to(device)
    # model.load_state_dict(torch.load('./weights/deep_ELU'))
    # test_loss, test_acc = test(model, test_loader)
    # print("deep_ELU",test_acc)
    