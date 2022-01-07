import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from ckip_transformers.nlp import CkipWordSegmenter

from utils import Preprocess, load_data, SimpleDatset
from models import LSTM_Net

# parameters
sen_len = 100
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 40
lr = 0.001
model_dir = 'model'
w2v_path = 'model/w2v_10.model'

def evaluation(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item()

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.CrossEntropyLoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            # print(type(labels[0]))
            loss = criterion(outputs, labels.long()) # 計算此時模型的 training loss
            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs) # 將 input 餵給模型
                outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = criterion(outputs, labels.long()) # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                if os.path.exists(f"{model_dir}/ckpt{int(best_acc/v_batch*1000)}.model"):
                    os.remove(f"{model_dir}/ckpt{int(best_acc/v_batch*1000)}.model")
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model.state_dict(), f"{model_dir}/ckpt{int(total_acc/v_batch*1000)}.model")
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
ws_driver = CkipWordSegmenter(device=0, level=3)
x, y = load_data('labels.csv', ws_driver)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
x = preprocess.sentence_word2idx()
y = torch.LongTensor(y)

# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=150, hidden_dim=80, output_dim=5, num_layers=1, dropout=0.4, fix_embedding=fix_embedding)
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# prepare data
dataset = SimpleDatset(X=x, y=y)
val_size = len(dataset)//8
train_size = len(dataset) - val_size
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
train_loader = data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
val_loader = data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)