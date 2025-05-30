import os
import torch
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import itertools
import csv

# ------------------------
# 全局参数配置
# ------------------------
class BaseConfig:
    """模型基础配置类，包含全局参数设置"""
    SEED = 42  # 随机种子，保证实验可复现
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 计算设备
    
    # 文件路径（示例路径，实际使用时需替换为具体路径）
    TRAIN_PATH = 'path/to/train.csv'  # 训练集CSV文件路径
    VAL_PATH = 'path/to/val.csv'      # 验证集CSV文件路径
    TEST_PATH = 'path/to/test.csv'    # 测试集CSV文件路径
    
    # 模型配置（示例路径，实际使用时需替换为具体模型路径）
    PRETRAINED_MODEL_PATH = 'path/to/pretrained_model'  # 预训练模型路径
    MODEL_SAVE_BASE_PATH = 'path/to/save_models/'       # 模型保存基础路径
    
    # 标签配置
    LABEL_NAMES = ['非攻击性', '攻击性']  # 分类标签名称
    
    # 数据预处理
    PREPROCESS = True  # 是否启用文本预处理

# 可调整的超参数网格
HYPER_PARAM_GRID = {
    'MAX_LEN': [64, 128, 256],          # 输入文本的最大长度
    'BATCH_SIZE': [16, 32, 64],         # 批次大小
    'LEARNING_RATE': [1e-5, 2e-5, 3e-5], # 学习率
    'EPOCHS': [3, 5, 10],              # 训练轮数
    'HIDDEN_DROPOUT_PROB': [0.1],      # 隐藏层Dropout概率
}

# 设置随机种子，确保实验结果可复现
def set_seed(seed=BaseConfig.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 打印设备信息
print(f"使用设备: {BaseConfig.DEVICE}")
if BaseConfig.DEVICE.type == "cuda":
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")

# 自定义数据集类，处理文本数据和标签
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        初始化微博数据集
        
        参数:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_len: 最大文本长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = BaseConfig.PREPROCESS
        
        # 对文本进行预处理（如果启用）
        if self.preprocess:
            self.texts = [self._preprocess(text) for text in self.texts]

    def _preprocess(self, text):
        """文本预处理函数，清理文本数据"""
        if not isinstance(text, str):
            text = str(text) if pd.notna(text) else ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # 去除URL
        text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
        text = re.sub(r'[^\w\s]', '', text)  # 去除特殊字符
        text = re.sub(r'\s+', ' ', text).strip()  # 规范化空格
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """获取单个样本，包括输入ID、注意力掩码、标签和原始文本"""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 使用分词器对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

# 注意力层，用于捕获文本中的重要信息
class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        """
        初始化注意力层
        
        参数:
            hidden_size: 隐藏层维度
        """
        super().__init__()
        self.attn = torch.nn.Linear(hidden_size, 1)  # 注意力权重计算
        self.softmax = torch.nn.Softmax(dim=1)       # 对注意力权重进行softmax归一化

    def forward(self, hidden_states):
        """
        前向传播计算
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            context: 上下文向量 [batch_size, hidden_size]
            attn_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        attn_weights = self.softmax(self.attn(hidden_states))  # [B, L, 1]
        context = torch.bmm(attn_weights.transpose(1, 2), hidden_states).squeeze(1)  # [B, H]
        return context, attn_weights

# MACBert分类模型，融合了预训练模型和注意力机制
class MACBertClassifier(torch.nn.Module):
    def __init__(self, hidden_dropout_prob):
        """
        初始化MACBert分类模型
        
        参数:
            hidden_dropout_prob: 隐藏层dropout概率
        """
        super().__init__()
        # 加载预训练的MACBert模型
        self.bert = AutoModel.from_pretrained(BaseConfig.PRETRAINED_MODEL_PATH)
        
        # 添加注意力层，增强模型对重要信息的关注
        self.attention = AttentionLayer(self.bert.config.hidden_size)
        
        # Dropout层和分类器
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(BaseConfig.LABEL_NAMES))

    def forward(self, input_ids, attention_mask):
        """
        前向传播计算
        
        参数:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        返回:
            logits: 分类得分 [batch_size, num_classes]
        """
        # 获取BERT模型的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # 应用注意力机制
        context, _ = self.attention(outputs.last_hidden_state)
        
        # 通过分类器得到最终分类得分
        logits = self.classifier(self.dropout(context))
        return logits

# 数据加载函数，用于加载和准备训练、验证和测试数据
def load_data(tokenizer, max_len, batch_size):
    """
    加载并准备数据集
    
    参数:
        tokenizer: 分词器
        max_len: 最大文本长度
        batch_size: 批次大小
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        tokenizer: 分词器
    """
    def _load_csv(path):
        """辅助函数：从CSV文件加载数据"""
        df = pd.read_csv(path)
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
    
    # 加载数据
    train_texts, train_labels = _load_csv(BaseConfig.TRAIN_PATH)
    val_texts, val_labels = _load_csv(BaseConfig.VAL_PATH)
    test_texts, test_labels = _load_csv(BaseConfig.TEST_PATH)
    
    # 创建数据集
    train_dataset = WeiboDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = WeiboDataset(val_texts, val_labels, tokenizer, max_len)
    test_dataset = WeiboDataset(test_texts, test_labels, tokenizer, max_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, tokenizer

# 模型评估函数，计算准确率、F1分数等指标
def evaluate(model, data_loader, stage="测试集"):
    """
    评估模型性能
    
    参数:
        model: 待评估的模型
        data_loader: 数据加载器
        stage: 评估阶段名称
        
    返回:
        acc: 准确率
        f1: F1分数
    """
    model.eval()  # 设置为评估模式
    y_true, y_pred = [], []
    
    with torch.no_grad(), tqdm(data_loader, desc=f"{stage}评估中") as pbar:
        for batch in pbar:
            # 将数据移至指定设备
            inputs = {k: v.to(BaseConfig.DEVICE) for k, v in batch.items() if k != 'text'}
            labels = inputs.pop('label')
            
            # 前向传播
            logits = model(**inputs)
            
            # 收集真实标签和预测标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=BaseConfig.LABEL_NAMES)
    
    print(f"\n{stage}评估结果:")
    print(f"准确率: {acc:.4f}, F1: {f1:.4f}")
    print("混淆矩阵:")
    print(pd.DataFrame(cm, index=BaseConfig.LABEL_NAMES, columns=BaseConfig.LABEL_NAMES))
    print("分类报告:\n", cr)
    
    return acc, f1

# 模型训练函数，使用指定参数训练模型
def train_model(params, tokenizer):
    """
    使用指定参数训练模型
    
    参数:
        params: 超参数字典
        tokenizer: 分词器
        
    返回:
        best_val_f1: 最佳验证集F1分数
        model_save_path: 最佳模型保存路径
    """
    # 创建保存路径
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d_%H%M")
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    model_save_path = f"{BaseConfig.MODEL_SAVE_BASE_PATH}model_{param_str}_{timestamp}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 加载数据
    train_loader, val_loader, _, _ = load_data(
        tokenizer, 
        max_len=params['MAX_LEN'], 
        batch_size=params['BATCH_SIZE']
    )
    
    # 初始化模型并移至指定设备
    model = MACBertClassifier(hidden_dropout_prob=params['HIDDEN_DROPOUT_PROB'])
    model.to(BaseConfig.DEVICE)
    
    # 配置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=params['LEARNING_RATE'], eps=1e-8)
    total_steps = len(train_loader) * params['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    
    # 早停机制配置
    best_val_f1 = 0.0
    early_stopping_patience = 2
    no_improvement_epochs = 0
    
    # 训练循环
    for epoch in range(params['EPOCHS']):
        print(f"\nEpoch {epoch+1}/{params['EPOCHS']}")
        print("-" * 20)
        
        # 训练阶段
        model.train()
        total_loss = 0
        y_true, y_pred = [], []
        
        with tqdm(train_loader, desc="训练中") as pbar:
            for batch in pbar:
                # 将数据移至指定设备
                inputs = {k: v.to(BaseConfig.DEVICE) for k, v in batch.items() if k != 'text'}
                labels = inputs.pop('label')
                
                # 前向传播和反向传播
                optimizer.zero_grad()
                logits = model(**inputs)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                optimizer.step()
                scheduler.step()
                
                # 收集训练信息
                total_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                
                pbar.set_postfix(loss=loss.item())
        
        # 计算训练指标
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"训练集 - 准确率: {train_acc:.4f}, F1: {train_f1:.4f}, 损失: {total_loss/len(train_loader):.4f}")
        
        # 验证阶段
        model.eval()
        total_loss = 0
        y_true_val, y_pred_val = [], []
        
        with torch.no_grad(), tqdm(val_loader, desc="验证中") as pbar:
            for batch in pbar:
                # 将数据移至指定设备
                inputs = {k: v.to(BaseConfig.DEVICE) for k, v in batch.items() if k != 'text'}
                labels = inputs.pop('label')
                
                # 前向传播
                logits = model(**inputs)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                
                # 收集验证信息
                total_loss += loss.item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(torch.argmax(logits, dim=1).cpu().numpy())
                
                pbar.set_postfix(loss=loss.item())
        
        # 计算验证指标
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')
        print(f"验证集 - 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, 损失: {total_loss/len(val_loader):.4f}")
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improvement_epochs = 0
            torch.save(model.state_dict(), model_save_path)  # 保存最佳模型
            print(f"保存最佳模型到: {model_save_path}")
        else:
            no_improvement_epochs += 1
            print(f"验证F1未提升，连续 {no_improvement_epochs}/{early_stopping_patience} 轮")
            if no_improvement_epochs >= early_stopping_patience:
                print("触发早停机制")
                break
    
    # 清理显存
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
    return best_val_f1, model_save_path

# 网格搜索函数，用于寻找最佳超参数组合
def grid_search():
    """
    执行超参数网格搜索
    
    返回:
        best_params: 最佳超参数组合
        best_model_path: 最佳模型路径
    """
    # 生成所有超参数组合
    param_combinations = []
    for params in itertools.product(*HYPER_PARAM_GRID.values()):
        param_dict = dict(zip(HYPER_PARAM_GRID.keys(), params))
        param_combinations.append(param_dict)
    
    total_combinations = len(param_combinations)
    print(f"网格搜索启动，共 {total_combinations} 组参数组合")
    
    # 创建结果存储文件
    result_file = f"{BaseConfig.MODEL_SAVE_BASE_PATH}grid_search_results_{datetime.now().strftime('%m%d_%H%M')}.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = list(HYPER_PARAM_GRID.keys()) + ['val_f1', 'model_path']
        writer.writerow(header)
    
    # 初始化tokenizer（避免重复加载）
    tokenizer = AutoTokenizer.from_pretrained(BaseConfig.PRETRAINED_MODEL_PATH)
    
    # 遍历所有组合
    best_f1 = 0.0
    best_params = {}
    best_model_path = ""
    
    for idx, params in enumerate(param_combinations):
        print(f"\n------------------- 第 {idx+1}/{total_combinations} 组参数 -------------------")
        print(f"参数: {params}")
        
        # 训练模型
        val_f1, model_path = train_model(params, tokenizer)
        
        # 记录结果
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = list(params.values()) + [val_f1, model_path]
            writer.writerow(row)
        
        # 更新最佳参数
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params
            best_model_path = model_path
            print(f"新的最佳验证F1: {best_f1:.4f}")
    
    print("\n------------------- 网格搜索结果 -------------------")
    print(f"最佳验证F1: {best_f1:.4f}")
    print(f"最佳参数组合: {best_params}")
    print(f"最佳模型路径: {best_model_path}")
    print(f"完整结果已保存至: {result_file}")
    
    return best_params, best_model_path

# 主函数，执行完整的训练、验证和测试流程
def main():
    """执行完整的模型训练、验证和测试流程"""
    # 执行网格搜索
    best_params, best_model_path = grid_search()
    
    # 使用最佳参数进行最终测试
    print("\n------------------- 使用最佳参数进行最终测试 -------------------")
    print(f"最佳参数: {best_params}")
    
    # 加载数据
    tokenizer = AutoTokenizer.from_pretrained(BaseConfig.PRETRAINED_MODEL_PATH)
    _, _, test_loader, _ = load_data(
        tokenizer, 
        max_len=best_params['MAX_LEN'], 
        batch_size=best_params['BATCH_SIZE']
    )
    
    # 初始化模型并加载最佳权重
    model = MACBertClassifier(hidden_dropout_prob=best_params['HIDDEN_DROPOUT_PROB'])
    model.load_state_dict(torch.load(best_model_path))
    model.to(BaseConfig.DEVICE)
    
    # 评估测试集
    evaluate(model, test_loader, stage="最终测试集")

if __name__ == "__main__":
    main()