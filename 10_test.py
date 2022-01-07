import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from ckip_transformers.nlp import CkipWordSegmenter

from utils import Preprocess, load_clean_data, load_data, SimpleDatset
from models import LSTM_Net

# parameters
sen_len = 100
fix_embedding = True # fix embedding during training
batch_size = 256
testdir = 'clean_corpus/ig_nonad.csv'
model_dir = 'model/lstm_w2v10_635.model'
w2v_path = 'model/w2v_10.model'
outputdir = 'output.csv'

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    ret_std = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            std = torch.std(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            ret_output += predicted.int().tolist()
            ret_std += std.float().tolist()
    
    return ret_output, ret_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading data ...") 
ws_driver = CkipWordSegmenter(device=0, level=3)
x, y = load_data(testdir, ws_driver)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
x = preprocess.sentence_word2idx()
y = torch.LongTensor(y)

# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=150, hidden_dim=80, output_dim=5, num_layers=1, dropout=0.4, fix_embedding=fix_embedding)
model.load_state_dict(torch.load(model_dir))
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# prepare data
testset = SimpleDatset(X=x, y=y)
test_loader = data.DataLoader(dataset = testset, batch_size = batch_size, shuffle = False, num_workers = 8)

# 開始訓練
preds_emoji, std_emoji = testing(batch_size, test_loader, model, device)

### testing emoji-less text ###

print("loading data ...") 
x, y = load_clean_data(testdir, ws_driver)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
x = preprocess.sentence_word2idx()
y = torch.LongTensor(y)

# prepare data
testset = SimpleDatset(X=x, y=y)
test_loader = data.DataLoader(dataset = testset, batch_size = batch_size, shuffle = False, num_workers = 8)

# 開始訓練
preds_emoless, std_emoless = testing(batch_size, test_loader, model, device)

output_df = pd.read_csv(testdir, index_col=[0])
output_df = output_df.drop(['label'], axis=1)
output_df['emoji label'] = preds_emoji
output_df['emoji std'] = std_emoji
output_df['emoji-less label'] = preds_emoless
output_df['emoji-less std'] = std_emoless
output_df.to_csv(outputdir)