import pandas as pd
from ResNet import BasicBlock, ResNet18, Bottleneck, ResNet50
from dataloader import LeukemiaLoader
import torch
from torch.utils.data import DataLoader
import opt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, evalDataset, batch_size):
    print("evaluate() not defined")
    loader = DataLoader(evalDataset, batch_size=batch_size)
    with torch.no_grad():
        model.eval()
        correct_pred = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            # print(y_pred.shape)
            y_pred_binary = torch.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)
            correct_pred += (y_pred_binary == y_batch).sum().item()
            loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
            # scheduler.step(loss)
        return loss, correct_pred / evalDataset.__len__()
        # print("accuracy: ", correct_pred / 1080)

def test(model, testDataset, batch_size):
    all_predictions = []
    print("test() not defined")
    loader = DataLoader(testDataset, batch_size=batch_size)
    with torch.no_grad():
        model.eval()
        for x_batch in loader:
            # print(x_batch.shape)
            x_batch = x_batch.to(device)
            
            y_pred = model(x_batch)
            y_pred_binary = torch.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)
            # correct_pred += (y_pred_binary == y_batch).sum().item()
            # loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
            # scheduler.step(loss)
            all_predictions.extend(y_pred_binary.cpu().tolist())
        return all_predictions
    

def train(model, trainDataset, evalDataset, learning_rate, epochs, batch_szie):
    print("train() not defined")
    trainLoader = DataLoader(trainDataset, batch_size=batch_size)
    
    model = model.to(device=device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
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
        train_acc = correct_pred / trainDataset.__len__()
        train_acc_list.append(train_acc)
        
        epoch_list.append(epoch)
        eval_loss, eval_acc = evaluate(model, evalDataset, batch_size)  
        eval_acc_list.append(eval_acc)
        
        model.train()
        scheduler.step(eval_loss)
         
        # if epoch % 20 == 0:
        print(f"##### epoch: {epoch} | {epochs} #####")
        print(f"train_loss: {loss.item()}, train_accuracy: {correct_pred / trainDataset.__len__()}")
        print("eval accuracy: ", eval_acc)    
    
    # plt.plot(epoch_list, train_acc_list, label=f"{activation}_train")    
    # plt.plot(epoch_list, test_acc_list, label=f"{activation}_test")

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./312553047_resnet18.csv", index=False)


if __name__ == "__main__":
    print("Good Luck :)")
    parser = opt.config_parser()
    args = parser.parse_args()
    
    mode = args.mode
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    test_csv_path = f"resnet_{args.ResnetModel}_test.csv"
    
    
    trainDataset = LeukemiaLoader("", "train")
    # trainLoader = trainLoader.to(device)
    valDataset = LeukemiaLoader("", "valid")
    # valLoader = valLoader.to(device)
    testDataset = LeukemiaLoader("", "test")
    
    if args.ResnetModel == "18":
        model = ResNet18(BasicBlock)
        model = model.to(device=device)
        # print(model)
    elif args.ResnetModel == "50":
        model = ResNet50(Bottleneck)
        model = model.to(device=device)
    
    if args.mode == "train":    
        train(model, trainDataset, valDataset, lr, epochs, batch_size)
        torch.save(model.state_dict(), f"./weights/1")
        predictions = test(model, testDataset, batch_size)
        print(predictions)
        save_result(test_csv_path, predictions)
        
    if args.mode == "test":
        model.load_state_dict(torch.load('./weights/1'))
        predictions = test(model, testDataset, batch_size)
        print(predictions)
        save_result(test_csv_path, predictions)


    
    