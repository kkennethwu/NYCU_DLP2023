import pandas as pd
from ResNet import BasicBlock, ResNet18
from dataloader import LeukemiaLoader
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


def evaluate(model, loader):
    print("evaluate() not defined")
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

def test():
    print("test() not defined")
    

def train(model, trainLoader, evalLoader, learning_rate, epochs):
    # print("train() not defined")
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True, eps=1e-4)
    # lossFn = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    eval_acc_list = []
    
    for epoch in range(1, epochs+1):
        correct_pred = 0
        model.train()
        for x_batch, y_batch in trainLoader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
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
        eval_loss, eval_acc = evaluate(model, evalLoader)  
        eval_acc_list.append(eval_acc)
        
        model.train()
        scheduler.step(eval_loss)
         
        if epoch % 20 == 0:
            print(f"##### epoch: {epoch} | {epochs} #####")
            print(f"train_loss: {loss.item()}, train_accuracy: {correct_pred / 1080}")
            print("eval accuracy: ", eval_acc)    
    
    # plt.plot(epoch_list, train_acc_list, label=f"{activation}_train")    
    # plt.plot(epoch_list, test_acc_list, label=f"{activation}_test")

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)


if __name__ == "__main__":
    print("Good Luck :)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    lr = 0.01
    epochs = 1000
    
    
    trainDataset = LeukemiaLoader("", "train")
    trainLoader = DataLoader(trainDataset, batch_size=batch_size)
    # trainLoader = trainLoader.to(device)
    valDataset = LeukemiaLoader("", "valid")
    valLoader = DataLoader(valDataset, batch_size=batch_size)
    # valLoader = valLoader.to(device)
    
    resNet18 = ResNet18(BasicBlock)
    train(resNet18, trainLoader, valLoader, lr, epochs)

    
    