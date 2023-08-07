import pandas as pd

def evaluate():
    print("evaluate() not defined")

def test():
    print("test() not defined")

def train():
    print("train() not defined")

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":
    print("Good Luck :)")