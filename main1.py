import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import jsonlines
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import torch.optim as optim


def load_glove_embeddings(glove_file, embedding_dim):
    """
    加载GloVe词嵌入文件，返回一个词嵌入字典和词汇表。
    :param glove_file: GloVe词嵌入文件路径
    :param embedding_dim: 词嵌入的维度
    :return: 词嵌入字典
    """
    word_embeddings = defaultdict(lambda: np.zeros(embedding_dim))  # 默认返回零向量
    word_embeddings['<UNK>'] = np.random.uniform(-0.1, 0.1, embedding_dim)  # 未知词

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                tokens = line.split()
                word = tokens[0]
                embedding = np.array(tokens[1:], dtype=np.float32)
                word_embeddings[word] = embedding
            except ValueError:
                # 如果无法转换为浮动数值，跳过该行
                continue
    return word_embeddings

def create_embedding_matrix(word_embeddings, vocab, embedding_dim):
    """
    使用词嵌入字典生成词嵌入矩阵。
    :param word_embeddings: 词嵌入字典
    :param vocab: 词汇表
    :param embedding_dim: 词嵌入的维度
    :return: 词嵌入矩阵
    """
    num_words = len(vocab)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i, word in tqdm(enumerate(vocab)):
        embedding_matrix[i] = word_embeddings[word]
    return torch.tensor(embedding_matrix, dtype=torch.float32)

def load_data(file_path):
    sentences = []
    labels = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            premise = obj['sentence1']
            hypothesis = obj['sentence2']
            label = obj['gold_label']
            
            if label == '-':  # 排除没有标签的句子对
                continue
            
            sentences.append((premise, hypothesis))
            labels.append(label)
    return sentences, labels

# 分词处理
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for premise, hypothesis in tqdm(sentences):
        tokenized_premise = word_tokenize(premise.lower())
        tokenized_hypothesis = word_tokenize(hypothesis.lower())
        tokenized_sentences.append((tokenized_premise, tokenized_hypothesis))
    return tokenized_sentences

# 创建词汇表
def build_vocab(tokenized_sentences):
    all_words = []
    for premise, hypothesis in tqdm(tokenized_sentences):
        all_words.extend(premise)
        all_words.extend(hypothesis)
    vocab = Counter(all_words)
    vocab = {word: idx+1 for idx, (word, _) in enumerate(vocab.most_common())}  # 索引从1开始
    vocab['<PAD>'] = 0  # 添加填充符
    return vocab

# 将句子转换为索引
def sentences_to_indices(tokenized_sentences, vocab, max_len=50):
    sentence_indices = []
    for premise, hypothesis in tqdm(tokenized_sentences):
        premise_indices = [vocab.get(word, vocab['<PAD>']) for word in premise]
        hypothesis_indices = [vocab.get(word, vocab['<PAD>']) for word in hypothesis]
        
        # 填充或截断句子
        premise_indices = premise_indices[:max_len] + [vocab['<PAD>']] * (max_len - len(premise_indices))
        hypothesis_indices = hypothesis_indices[:max_len] + [vocab['<PAD>']] * (max_len - len(hypothesis_indices))
        
        sentence_indices.append((torch.tensor(premise_indices), torch.tensor(hypothesis_indices)))
    return sentence_indices


# 2. 数据加载器

def create_dataloader(sentences, labels, vocab, batch_size=32):
    # 将句子转换为索引
    tokenized_sentences = tokenize_sentences(sentences)
    sentence_indices = sentences_to_indices(tokenized_sentences, vocab)
    
    # 将标签转为数字（Entailment -> 0, Neutral -> 1, Contradiction -> 2）
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    labels = [label_map[label] for label in labels]
    
    # 创建TensorDataset
    premise_data = torch.stack([pair[0] for pair in sentence_indices])
    hypothesis_data = torch.stack([pair[1] for pair in sentence_indices])
    label_data = torch.tensor(labels)
    
    dataset = TensorDataset(premise_data, hypothesis_data, label_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 定义模型
class InputEncodingLayer(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super(InputEncodingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.bilstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, premise, hypothesis):
        premise_embedded = self.embedding(premise)
        hypothesis_embedded = self.embedding(hypothesis)
        premise_encoded, _ = self.bilstm(premise_embedded)
        hypothesis_encoded, _ = self.bilstm(hypothesis_embedded)
        return premise_encoded, hypothesis_encoded

class LocalInferenceLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LocalInferenceLayer, self).__init__()

    def forward(self, premise_encoded, hypothesis_encoded):
        eij = self.calculate_attention(premise_encoded, hypothesis_encoded)
        a_tilde = self.weighted_sum(eij, hypothesis_encoded)
        b_tilde = self.weighted_sum(eij.transpose(1, 2), premise_encoded)
        return a_tilde, b_tilde
    
    def calculate_attention(self, premise_encoded, hypothesis_encoded):
        premise_norm = premise_encoded / premise_encoded.norm(p=2, dim=2, keepdim=True)
        hypothesis_norm = hypothesis_encoded / hypothesis_encoded.norm(p=2, dim=2, keepdim=True)
        eij = torch.bmm(premise_norm, hypothesis_norm.transpose(1, 2))
        return eij
    
    def weighted_sum(self, attention_weights, encoded_sequence):
        attention_weights = F.softmax(attention_weights, dim=-1)
        weighted_sum = torch.bmm(attention_weights, encoded_sequence)
        return weighted_sum

class InferenceCompositionLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(InferenceCompositionLayer, self).__init__()
        self.bilstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, a_tilde, b_tilde):
        combined = torch.cat((a_tilde, b_tilde), dim=1)
        lstm_out, _ = self.bilstm(combined)
        max_pooling = torch.max(lstm_out, dim=1)[0]
        avg_pooling = torch.mean(lstm_out, dim=1)
        pooled_output = torch.cat((max_pooling, avg_pooling), dim=1)
        output = self.fc(pooled_output)
        return output

# 损失函数和优化器
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for premise, hypothesis, labels in tqdm(train_loader):
        premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(premise, hypothesis)

        # 计算损失
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for premise, hypothesis, labels in tqdm(val_loader):
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)

            # 前向传播
            output = model(premise, hypothesis)

            # 计算损失
            loss = criterion(output, labels)
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

class ESIMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(ESIMModel, self).__init__()
        self.input_encoding_layer = InputEncodingLayer(embedding_matrix, hidden_dim)
        self.local_inference_layer = LocalInferenceLayer(hidden_dim)
        self.inference_composition_layer = InferenceCompositionLayer(hidden_dim, output_dim)

    def forward(self, premise, hypothesis):
        # Step 1: Encoding the premise and hypothesis
        premise_encoded, hypothesis_encoded = self.input_encoding_layer(premise, hypothesis)
        
        # Step 2: Local inference modeling
        a_tilde, b_tilde = self.local_inference_layer(premise_encoded, hypothesis_encoded)
        
        # Step 3: Inference composition
        output = self.inference_composition_layer(a_tilde, b_tilde)
        
        return output
def predict(model,test_loader,device):
    correct=0
    total=0
    for premise,hypothesis, labels in tqdm(test_loader):
        premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
        output=model(premise,hypothesis)
        _,predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item() 
        total += labels.size(0) 
        
    accuracy = correct/total 
    print(f"Accuracy: {accuracy:.4f}")
    

    
def main():
    train_file = '/Users/zzy/Desktop/nlpbeginner/project3/snli_1.0/train.jsonl'
    val_file = '/Users/zzy/Desktop/nlpbeginner/project3/snli_1.0/val.jsonl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_sentences, train_labels = load_data(train_file)
    val_sentences, val_labels = load_data(val_file)
    # 构建词汇表
    tokenized_train_sentences = tokenize_sentences(train_sentences)
    vocab = build_vocab(tokenized_train_sentences)
    glove_file = '/Users/zzy/Desktop/nlpbeginner/ESIM/glove.840B.300d.txt'  # 这里假设你已经下载了300维的GloVe词嵌入
    embedding_dim = 300  # GloVe的300维词嵌入

    word_embeddings = load_glove_embeddings(glove_file, embedding_dim)
    embedding_matrix = create_embedding_matrix(word_embeddings, vocab, embedding_dim).to(device)

    # 检查词嵌入矩阵的形状
    print(embedding_matrix.shape)  # 应该是 (词汇表大小, 300)
    # 创建DataLoader
    train_loader = create_dataloader(train_sentences, train_labels, vocab, batch_size=32)
    val_loader = create_dataloader(val_sentences, val_labels, vocab, batch_size=32)
    batch_size=32
    hidden_dim=256 
    output_dim=3
    model = ESIMModel(embedding_matrix,hidden_dim,output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
    test_file = '/Users/zzy/Desktop/nlpbeginner/project3/snli_1.0/val.jsonl'
    test_sentences, test_labels = load_data(train_file)
    test_loader = create_dataloader(test_sentences, test_labels, vocab, batch_size=32)
    predict(model,test_loader,device)

if __name__ == "__main__":
    main()


    
    

