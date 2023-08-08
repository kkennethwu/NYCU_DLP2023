import pandas as pd
from ResNet import BasicBlock, ResNet18, Bottleneck, ResNet50, ResNet152
from dataloader import LeukemiaLoader
import torch
from torch.utils.data import DataLoader
import opt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, evalDataset, batch_size):
    print("evaluate() not defined")
    all_predictions = []
    all_label = []
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
            all_predictions.extend(y_pred_binary.cpu().tolist())
            all_label.extend(y_batch.cpu().tolist())
        confusion = confusion_matrix(all_label, all_predictions)
        return confusion, loss, correct_pred / evalDataset.__len__()
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
    

def train(model, trainDataset, evalDataset, learning_rate, epochs, batch_size, model_name):
    print("train() not defined")
    trainLoader = DataLoader(trainDataset, batch_size=batch_size)
    
    model = model.to(device=device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.875, weight_decay=3.0517578125e-05)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
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
        _, eval_loss, eval_acc = evaluate(model, evalDataset, batch_size)  
        eval_acc_list.append(eval_acc)
        
        model.train()
        scheduler.step()
         
        # if epoch % 20 == 0:
        print(f"##### epoch: {epoch} | {epochs} #####")
        print(f"train_loss: {loss.item()}, train_accuracy: {correct_pred / trainDataset.__len__()}")
        print("eval accuracy: ", eval_acc)    
    
    plt.plot(epoch_list, train_acc_list, label=f"ResNet{model_name}_train")    
    plt.plot(epoch_list, eval_acc_list, label=f"ResNet{model_name}_test")
    plt.savefig("./Comparison figures")

def save_result(csv_path, predict_result, ResnetModel):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(f"./312553047_resnet{ResnetModel}.csv", index=False)
    
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")



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
    testDataset = LeukemiaLoader("", "test", args.ResnetModel)
    
    
    
    if args.compare_figure == True:
        model = ResNet18(BasicBlock)
        model = model.to(device=device)
        train(model, trainDataset, valDataset, lr, epochs, batch_size, 18)
        model = ResNet50(Bottleneck)
        model = model.to(device=device)
        train(model, trainDataset, valDataset, lr, epochs, batch_size, 50)
        model = ResNet152(Bottleneck)
        model = model.to(device=device)
        train(model, trainDataset, valDataset, lr, epochs, batch_size, 152)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig("./Comparison figures")
        
    else:
        if args.ResnetModel == "18":
            model = ResNet18(BasicBlock)
            model = model.to(device=device)
            # print(model)
        elif args.ResnetModel == "50":
            model = ResNet50(Bottleneck)
            model = model.to(device=device)
            # print(model)
        elif args.ResnetModel == "152":
            model = ResNet152(Bottleneck)
            model = model.to(device=device)
            # print(model)
    
        if args.mode == "train":    
            train(model, trainDataset, valDataset, lr, epochs, batch_size, args.ResnetModel)
            torch.save(model.state_dict(), f"./weights/1")
            predictions = test(model, testDataset, batch_size)
            print(predictions)
            save_result(test_csv_path, predictions, args.ResnetModel)
            
        if args.mode == "test":
            model.load_state_dict(torch.load(args.model_weight_path))
            predictions = test(model, testDataset, batch_size)
            print(predictions)
            save_result(test_csv_path, predictions, args.ResnetModel)

        if args.mode == "eval":
            model.load_state_dict(torch.load(args.model_weight_path))
            confusion, loss, accuracy = evaluate(model, valDataset, batch_size)     
            print(accuracy)   
            # Define your class labels
            class_labels = ["0", "1"]
            # Plot the confusion matrix
            plot_confusion_matrix(confusion, class_labels, normalize=True)
            plt.savefig("./confusion_matrix")
        

    
    