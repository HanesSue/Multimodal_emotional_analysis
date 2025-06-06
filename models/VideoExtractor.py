import os
import tempfile
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import torch
import numpy as np
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from utils.SIMSData import SIMSData
import math
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.VideosProcessor import VideosProcessor
class SIMSVideoDataset(Dataset):
    def __init__(self, sims_data, feature_dict, transform=None, num_frames=144):
        """
        参数:
            sims_data: SIMSData实例，包含视频路径和标签
            feature_dict: 从pkl文件加载的特征字典，key为"video_id_clip_id"
            transform: 图像预处理转换
            num_frames: 每个视频采样的帧数
        """
        self.sims_data = sims_data
        self.feature_dict = feature_dict
        self.num_frames = num_frames

    def __len__(self):
        return len(self.sims_data)

    def __getitem__(self, idx):
        video_info, _, labels = self.sims_data[idx]  # video_info 是文件路径字符串
    
        # 解析 video_id 和 clip_id
        filename_with_ext = os.path.basename(video_info)
        clip_id = os.path.splitext(filename_with_ext)[0]
        parent_dir = os.path.dirname(video_info)
        video_id = os.path.basename(parent_dir)


        # 获取特征向量
        # 构建特征字典中的键
        feature_key = f"{video_id}_{clip_id}"
        
        # 获取对应的特征向量
        if feature_key in self.feature_dict:
            features = self.feature_dict[feature_key]['vision']
        else:
            print(f"警告: 未找到特征向量 for {feature_key}")
            # 返回全零特征或抛出异常，取决于您的处理逻辑
            features = np.zeros((self.num_frames, 35))  # 假设特征维度为709
        
        # 特征采样（如果帧数超过num_frames）
        if len(features) > self.num_frames:
            # 均匀采样帧
            indices = np.linspace(0, len(features)-1, self.num_frames, dtype=int)
            features = features[indices]
        elif len(features) < self.num_frames:
            # 不足则填充零帧
            pad_len = self.num_frames - len(features)
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        
        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        
        # 获取情感标签
        label_V = torch.tensor(labels[3], dtype=torch.float)  # valence维度
        
        return features, label_V

class SpecialSomoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction, target):
        error = torch.abs(prediction - target)
        loss = torch.zeros_like(error)
        # 区间一：（0 ≤ error < 0.4）
        cond_f = error < 0.4
        loss[cond_f] = 1.25 * error[cond_f].pow(2)

        # 区间二：（0.4 ≤ error < 1.0）
        cond_g = (error >= 0.4) & (error < 1.0)
        loss[cond_g] = error[cond_g] - 0.2

        # 区间三：（error ≥ 1.0），不希望为 0，否则极端误差没有惩罚
        cond_h = error >= 1.0
        loss[cond_h] = error[cond_h] - 0.2   # 或者你想要更大的惩罚，比如 error[cond_h] - 0.2 + (error[cond_h] - 1.0)

        return loss.mean()


class VideoExtractor:
    """
    视频情感分析器：
    参数 type:   "train" (训练新的模型)  "load" (加载已经训练好的模型)
    """
    def __init__(self, type="train", details=True, **kwargs):
        self.type = type
        self.details = details
        self.DEVICE = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.NUM_FRAMES = kwargs.get("num_frames", 144)
        self.BATCH_SIZE = kwargs.get("batch_size", 64)
        self.EPOCHS = kwargs.get("epochs", 40)
        self.PATIENCE = kwargs.get("patience", 10)
        self.LEARNING_RATE = kwargs.get("learning_rate", 1e-3)
        self.MODEL_NAME = kwargs.get("model_name", "emotion_regressor")
        self.MODEL_DIR = kwargs.get("model_dir", "saved_model")
        self.MODEL_SAVE_PATH = os.path.join(self.MODEL_DIR, f"{self.MODEL_NAME}.pth")
        self.DATA_ROOT = kwargs.get("data_root", "data/ch-sims2s/ch-simsv2s")
        self.PKL_ROOT = kwargs.get("pkl_root", "data/ch-sims2s")
        self.LR_SCHEDULER_FACTOR = kwargs.get("lr_scheduler_factor", 0.5)
        self.LR_SCHEDULER_PATIENCE = kwargs.get("lr_scheduler_patience", 5)
        # 标签配置
        self.DISCRETE_VALUES = [
            -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1
        ]  # 离散化标签  

        # 创建模型目录
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
        if details:
            print("正在初始化视频情感分析器...")
        
        # 加载OpenFace特征
        self.load_openface_features()
        
        # 初始化数据集
        self.init_datasets()
        
        # 根据类型初始化模型
        if self.type == "train":
            self.init_model_for_training()
        elif self.type == "load":
            self.load_trained_model()
        else:
            raise ValueError("type参数必须为'train'或'load'")
    
    def load_openface_features(self):
        """加载OpenFace提取的面部特征"""
        try:
            pkl_path = os.path.join(self.PKL_ROOT, "openface_features_AU.pkl")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.features_dict = data["videos"]
                print(f"✅ 成功加载{len(self.features_dict)}个视频的特征")
        except Exception as e:
            print(f"❌ 加载特征失败: {e}")
            self.features_dict = {}
    
    def init_datasets(self):
        """初始化训练、验证和测试数据集（含分桶级别均衡采样）"""
        if self.details:
            print("正在初始化数据集...")

        # 1. 加载原始数据
        self.train_raw = SIMSData(root=self.DATA_ROOT, mode="train")
        self.valid_raw = SIMSData(root=self.DATA_ROOT, mode="valid")
        self.test_raw  = SIMSData(root=self.DATA_ROOT, mode="test")

        # 2. 创建视频特征数据集
        self.train_dataset = SIMSVideoDataset(
            self.train_raw,
            self.features_dict,
            num_frames=self.NUM_FRAMES
        )
        self.valid_dataset = SIMSVideoDataset(
            self.valid_raw,
            self.features_dict,
            num_frames=self.NUM_FRAMES
        )
        self.test_dataset  = SIMSVideoDataset(
            self.test_raw,
            self.features_dict,
            num_frames=self.NUM_FRAMES
        )

        # 3. 统计训练集每条样本的离散标签（label_V）以及各标签的计数
        #    SIMSVideoDataset 的 __getitem__ 返回 (features, label_V)，其中 label_V 是一个 0-11 之间的离散值
        all_labels = []
        for i in range(len(self.train_dataset)):
            _, label_V = self.train_dataset[i]  # 这里 label_V 是一个 Tensor
            raw_val = label_V.item()                    # 从 Tensor 里取出 Python float
            disc_val = self.discretize(raw_val)         # 传入 float，而不是 Tensor
            all_labels.append(disc_val)

        # 统计每个 label_V 出现的次数
        counter = Counter(all_labels)
        class_to_weight = {}
        for val in self.DISCRETE_VALUES:
            # 如果某个值在 counter 里不存在（count=0），也给一个很小的 weight 防止除 0
            count_val = counter.get(val, 0)
            class_to_weight[val] = 1.0 / (count_val + 1e-6)

        # 5. 为训练集中每条样本分配“对应标签的权重”
        sample_weights = [ class_to_weight[label] for label in all_labels ]  # 长度 = len(train_dataset)
        # 6. 用 sample_weights 创建 WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,               # 每条样本的采样权重列表
            num_samples=len(sample_weights),      # 每个 epoch 抽取样本总数，可以设置为 len(sample_weights)
            replacement=True                      # 有放回抽样
        )

        # 7. 用这个 sampler 替换原来的 shuffle=True，创建 train_loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE,
            sampler=sampler,       
            num_workers=0,
            pin_memory=True
        )

        # 验证集和测试集仍按原先方式（不做均衡采样）
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        if self.details:
            print(f"训练集: {len(self.train_dataset)} samples")
            print (f"训练集分布: {counter}")
            print( f"训练集权重: {class_to_weight}")
            print(f"验证集: {len(self.valid_dataset)} samples")
            print(f"测试集: {len(self.test_dataset)} samples")
    def init_model_for_training(self):
        """初始化用于训练的模型"""
        self.model = self._create_model()
        self.model.to(self.DEVICE)
        
        # 定义损失函数和优化器
        #self.criterion = nn.MSELoss()
        self.criterion = SpecialSomoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.LEARNING_RATE
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',          # 监控验证损失（越小越好）
            factor=self.LR_SCHEDULER_FACTOR,  # 学习率衰减因子（如0.5表示减半）
            patience=self.LR_SCHEDULER_PATIENCE,  # 连续多少个epoch未改善则衰减学习率
            verbose=True,       # 打印学习率调整信息
            min_lr=1e-7         # 最小学习率，防止学习率过低
        )
        
        if self.details:
            print("模型初始化完成，准备训练...")
            print(f"学习率调度器已设置：每{self.LR_SCHEDULER_PATIENCE}个epoch未改善则衰减至{self.LR_SCHEDULER_FACTOR}倍")
    
    def _create_model(self):
        """创建视频情感回归模型"""    
        class EmotionRegressor(nn.Module):
            def __init__(self, input_dim=35, hidden_dim=128):
                super(EmotionRegressor, self).__init__()
                
                # 特征提取层
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )
                
                # 时序建模
                self.lstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3
                )
                
                # 回归输出层
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1)  # 预测valence值
                )
            
            def forward(self, x):
                batch_size = x.size(0)
                
                # 特征提取
                x = self.feature_extractor(x)
                
                # LSTM处理时序信息
                lstm_out, _ = self.lstm(x)
                
                # 使用最后一个时间步的输出
                last_output = lstm_out[:, -1, :]
                
                # 回归预测
                output = self.regressor(last_output)
                
                return output
        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim=35, d_model=256, nhead=4, num_layers=3, dropout=0.3):
                super(TransformerRegressor, self).__init__()

                # 第1步：将原始特征投到 d_model 维度
                self.input_fc = nn.Linear(input_dim, d_model)

                # Transformer EncoderLayer，堆叠 num_layers 个
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation='relu',
                    batch_first=True   # 直接使用 [B, T, d_model]
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # 回归输出：先做平均池化，再全连接
                self.output_fc = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

            def forward(self, x):
                """
                x: [B, T, input_dim]
                """
                # 1) 投到 Transformer 所需维度
                x_proj = self.input_fc(x)  # [B, T, d_model]

                # 2) 一般不做 position encoding（因为 T 比较小），也可以加 learnable PE
                #    这里直接套用原始 x_proj
                #    transformer expects [B, T, d_model] if batch_first=True
                out = self.transformer(x_proj)  # [B, T, d_model]

                # 3) 对时间维度做平均池化
                pooled = out.mean(dim=1)       # [B, d_model]

                # 4) 回归输出
                output = self.output_fc(pooled)  # [B, 1]
                return output
        class PositionalEncoding(nn.Module):
            """标准的正弦/余弦位置编码，用于给 Transformer 提供时序信息。"""
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                # pe: [1, max_len, d_model]
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
                )  # [d_model/2]
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)  # [1, max_len, d_model]
                self.register_buffer("pe", pe)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: [B, T, d_model]
                Returns:
                    x + pe[:, :T, :]，形状不变
                """
                seq_len = x.size(1)
                return x + self.pe[:, :seq_len, :]
        class WithBiasConvTransformerRegressor(nn.Module):
            """
            先用 1D 卷积在时间维度上做局部特征提取，再把卷积输出序列喂给 TransformerEncoder，
            最后做平均池化接全连接回归输出单一连续值（valence）。
            """
            def __init__(
                self,
                input_dim: int = 35,
                conv_channels: int = 64,
                conv_kernel: int = 3,
                d_model: int = 128,
                nhead: int = 4,
                num_layers: int = 2,
                dropout: float = 0.3,
            ):
                super().__init__()
                # ---------------------------------------------------------
                # 第一阶段：1D 卷积
                #   输入: [B, T, input_dim] → permute → [B, input_dim, T]
                #   输出: [B, conv_channels, T] → permute → [B, T, conv_channels]
                # ---------------------------------------------------------
                padding = conv_kernel // 2  # 保证输出时序长度和输入相同
                self.conv_block = nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                            kernel_size=conv_kernel, padding=padding),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),

                    # 你可以自行叠加更多 Conv1d 层：
                    # nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel, padding=padding),
                    # nn.BatchNorm1d(conv_channels),
                    # nn.ReLU(inplace=True),
                    # nn.Dropout(dropout),
                )

                # ---------------------------------------------------------
                # 第二阶段：Transformer Encoder
                #   输入: [B, T, conv_channels] → 先线性投到 d_model → 添加位置编码 → Transformer
                #   输出: [B, T, d_model]
                # ---------------------------------------------------------
                self.input_fc = nn.Linear(conv_channels, d_model)
                self.pos_enc = PositionalEncoding(d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="relu",
                    batch_first=True,  # 让输入为 [B, T, d_model]
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # ---------------------------------------------------------
                # 第三阶段：回归输出层
                #   对 Transformer 输出在时间维度做平均池化 → [B, d_model] → 全连接 → 单值回归
                # ---------------------------------------------------------

                self.norm = nn.LayerNorm(d_model) 
                self.regressor = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),  # 预测 valence
                )
                # **新增一个偏置参数（标量）**
                self.output_bias = nn.Parameter(torch.zeros(1))  # 初始化为 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: [B, T, input_dim]，input_dim=35
                Returns:
                    output: [B]（预测的连续 valence 值，后面可 .squeeze()）
                """
                # 1) 1D 卷积阶段
                # x 原本 [B, T, input_dim]，Conv1d 需要 [B, C_in, T] → permute
                x_conv = x.permute(0, 2, 1)       # [B, input_dim, T]
                c = self.conv_block(x_conv)      # [B, conv_channels, T]
                c = c.permute(0, 2, 1)            # [B, T, conv_channels]

                # 2) 线性投到 d_model + 位置编码
                c_proj = self.input_fc(c)        # [B, T, d_model]
                c_pe   = self.pos_enc(c_proj)    # [B, T, d_model]

                # 3) Transformer Encoder
                out = self.transformer(c_pe)     # [B, T, d_model]

                # 4) 时间维度平均池化
                pooled = out.mean(dim=1)         # [B, d_model]

                # 5) 回归输出
                cont_pred = self.regressor(pooled).squeeze(-1)  # [B]
                # 在标准化空间里加上 bias
                cont_pred = cont_pred + self.output_bias       # [B]
                return cont_pred          # [B]    
        class ConvTransformerRegressor(nn.Module):
            """
            先用 1D 卷积在时间维度上做局部特征提取，再把卷积输出序列喂给 TransformerEncoder，
            最后做平均池化接全连接回归输出单一连续值（valence）。
            """
            def __init__(
                self,
                input_dim: int = 35,
                conv_channels: int = 64,
                conv_kernel: int = 3,
                d_model: int = 128,
                nhead: int = 4,
                num_layers: int = 2,
                dropout: float = 0.3,
            ):
                super().__init__()
                # ---------------------------------------------------------
                # 第一阶段：1D 卷积
                #   输入: [B, T, input_dim] → permute → [B, input_dim, T]
                #   输出: [B, conv_channels, T] → permute → [B, T, conv_channels]
                # ---------------------------------------------------------
                padding = conv_kernel // 2  # 保证输出时序长度和输入相同
                self.conv_block = nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                            kernel_size=conv_kernel, padding=padding),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),

                    # 你可以自行叠加更多 Conv1d 层：
                    # nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel, padding=padding),
                    # nn.BatchNorm1d(conv_channels),
                    # nn.ReLU(inplace=True),
                    # nn.Dropout(dropout),
                )

                # ---------------------------------------------------------
                # 第二阶段：Transformer Encoder
                #   输入: [B, T, conv_channels] → 先线性投到 d_model → 添加位置编码 → Transformer
                #   输出: [B, T, d_model]
                # ---------------------------------------------------------
                self.input_fc = nn.Linear(conv_channels, d_model)
                self.pos_enc = PositionalEncoding(d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="relu",
                    batch_first=True,  # 让输入为 [B, T, d_model]
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # ---------------------------------------------------------
                # 第三阶段：回归输出层
                #   对 Transformer 输出在时间维度做平均池化 → [B, d_model] → 全连接 → 单值回归
                # ---------------------------------------------------------
                self.regressor = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),  # 预测 valence
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: [B, T, input_dim]，input_dim=35
                Returns:
                    output: [B]（预测的连续 valence 值，后面可 .squeeze()）
                """
                # 1) 1D 卷积阶段
                # x 原本 [B, T, input_dim]，Conv1d 需要 [B, C_in, T] → permute
                x_conv = x.permute(0, 2, 1)       # [B, input_dim, T]
                c = self.conv_block(x_conv)      # [B, conv_channels, T]
                c = c.permute(0, 2, 1)            # [B, T, conv_channels]

                # 2) 线性投到 d_model + 位置编码
                c_proj = self.input_fc(c)        # [B, T, d_model]
                c_pe   = self.pos_enc(c_proj)    # [B, T, d_model]

                # 3) Transformer Encoder
                out = self.transformer(c_pe)     # [B, T, d_model]

                # 4) 时间维度平均池化
                pooled = out.mean(dim=1)         # [B, d_model]

                # 5) 回归输出
                pred = self.regressor(pooled)    # [B, 1]
                return pred.squeeze(-1)          # [B]    
        return WithBiasConvTransformerRegressor()
    def load_trained_model(self):
        """加载预训练模型"""
        if os.path.exists(self.MODEL_SAVE_PATH):
            self.model = self._create_model()
            self.model.load_state_dict(torch.load(self.MODEL_SAVE_PATH))
            self.model.to(self.DEVICE)
            self.model.eval()
            if self.details:
                print(f"✅ 已加载预训练模型: {self.MODEL_SAVE_PATH}")
        else:
            raise FileNotFoundError(f"未找到模型文件: {self.MODEL_SAVE_PATH}")
    
    def train(self):
        """训练情感回归模型"""
        if self.type != "train":
            raise ValueError("模型未配置为训练模式")
        
        if self.details:
            print("开始训练模型...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.EPOCHS):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.EPOCHS}") as pbar:
                for features, labels in self.train_loader:
                    features = features.to(self.DEVICE)
                    labels = labels.to(self.DEVICE)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(features).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with tqdm(total=len(self.valid_loader), desc="Validation") as pbar:
                with torch.no_grad():
                    for features, labels in self.valid_loader:
                        features = features.to(self.DEVICE)
                        labels = labels.to(self.DEVICE)
                        
                        outputs = self.model(features).squeeze()
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        pbar.update(1)
                        pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
            
            # 计算平均损失
            train_loss /= len(self.train_loader)
            val_loss /= len(self.valid_loader)
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)  # 传入验证损失以调整学习率
            
            # 打印训练进度（包含当前学习率）
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.details:
                print(f"\nEpoch {epoch+1}/{self.EPOCHS} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.MODEL_SAVE_PATH)
                patience_counter = 0
                if self.details:
                    print(f"✅ 模型保存: 验证损失改善 ({best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.PATIENCE:
                    if self.details:
                        print(f"🛑 早停: 验证损失在{self.PATIENCE}个周期内未改善")
                    break
        
        if self.details:
            print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
            print(f"最终学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
    


    def test_model(self):
        """加载模型并在测试集上运行评估"""
        if self.type != "load":
            self.load_trained_model()  # 自动切换为加载模式
        
        # 确保模型处于评估模式
        self.model.eval()
        # criterion = SpecialSomoothL1Loss()
        
        # 初始化评估指标
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # 进度条显示
        progress = tqdm(self.test_loader, desc="Testing", unit="batch")
        with torch.no_grad():
            for features, labels in progress:
                features = features.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                
                # 模型预测
                outputs = self.model(features).squeeze()
                # loss = criterion(outputs, labels).item()
                # total_loss += loss * len(labels)
                
                # 收集结果
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算连续指标
        # avg_loss = total_loss / len(self.test_dataset)
        mse = mean_squared_error(all_labels, all_predictions)
        rmse = np.sqrt(mse)
        discrete_labels = [self.discretize(l) for l in all_labels]
        discrete_predictions = [self.discretize(s) for s in all_predictions]
        
        evaluation = self.evaluate(y_true=discrete_labels, y_pred=discrete_predictions)
        acc5 = evaluation["ACC5"]
        acc2 = evaluation["ACC2"]
        f1 = evaluation["F1"]
        
        # 计算分类指标
        
        # 打印结果
        print("\n================= 测试集评估结果 =================")
        print(f"平均损失 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"五分类准确率 (ACC5): {acc5:.4f}")
        print(f"二分类准确率 (ACC2): {acc2:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"测试集样本数: {len(all_labels)}")
        print("================================================\n")
        
        

        return {
            "mse": mse,
            "rmse": rmse,
            "acc5": acc5,
            "acc2": acc2,
            "sample_count": len(all_labels)
        }
    
    
    def predict(self, video_path, openface_exe_path):
        """
        对给定的视频进行情感预测
        
        参数:
            video_path (str): 输入视频文件路径
            openface_exe_path (str): OpenFace可执行文件路径
            
        返回:
            Dict[str, Any]: 包含情感预测结果的字典
        """
        # 检查模型是否已加载
        if self.type != "load" or not hasattr(self, 'model') or self.model is None:
            self.load_trained_model()
        
        self.model.eval()

        if not openface_exe_path or not os.path.exists(openface_exe_path):
            raise FileNotFoundError(f"OpenFace可执行文件不存在: {openface_exe_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: 使用 OpenFace 提取特征
            print(f"📹 正在处理视频: {video_path}")
            video_processor = VideosProcessor(openface_exe_path)
            success_count = video_processor.process_videos(
                input_path=video_path,
                output_dir=temp_dir
            )
            if success_count == 0:
                raise RuntimeError("❌ 无法处理视频或未生成有效特征")

            # Step 2: 提取和归一化特征
            print("🧪 正在提取和归一化特征...")
            features = video_processor.extract_and_normalize_features(
                csv_folder=temp_dir,
                output_pkl_path=os.path.join(temp_dir, "features.pkl")
            )

            # Step 3: 预处理为模型输入格式
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            if video_id not in features["videos"]:
                raise RuntimeError(f"⚠️ 未找到视频 {video_id} 的有效特征")

            vision_features = features["videos"][video_id]["vision"]  # shape: [T, D]
            vision_features = np.array(vision_features)

            # 对帧数进行裁剪或填充
            num_frames = self.NUM_FRAMES
            if vision_features.shape[0] >= num_frames:
                vision_features = vision_features[:num_frames]
            else:
                pad_len = num_frames - vision_features.shape[0]
                pad = np.zeros((pad_len, vision_features.shape[1]))
                vision_features = np.concatenate((vision_features, pad), axis=0)

            input_tensor = torch.tensor(vision_features, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)  # shape: [1, T, D]

            # Step 4: 情感预测
            with torch.no_grad():
                output = self.model(input_tensor).squeeze().item()  # 单个标量
                discrete_output = self.discretize(output)

            # Step 5: 返回结果
            return {
                "continuous": round(output, 4),
                "discrete_label": self.score_to_label(discrete_output),
                "discrete_value": discrete_output  
            }

    def discretize(self, value):
        """将连续值离散化为预定义的离散值"""
        values = np.array(self.DISCRETE_VALUES)
        return  float(values[np.argmin(np.abs(values - value))]) 

        
    def score_to_label(self,score: float) -> str:
        TOLERANCE = 1e-6
        def is_close(a, b):
            return abs(a - b) < TOLERANCE
        if is_close(score, -1.0) or is_close(score, -0.8):
            return '负'
        elif is_close(score, -0.6) or is_close(score, -0.4) or is_close(score, -0.2):
            return '弱负'
        elif is_close(score, 0.0):
            return '中'
        elif is_close(score, 0.2) or is_close(score, 0.4) or is_close(score, 0.6):
            return '弱正'
        elif is_close(score, 0.8) or is_close(score, 1.0):
            return '正'
        else:
            raise ValueError(f"Unexpected sentiment score: {score}")
        
    def label_to_acc2(self,label: str) -> int:
    # 负/弱负 -> 0, 中/弱正/正 -> 1
        return 0 if label in ['负', '弱负'] else 1

    def evaluate(self,y_true, y_pred):
        # 转为5类标签
        y_true_5 = [self.score_to_label(s) for s in y_true]
        y_pred_5 = [self.score_to_label(s) for s in y_pred]
        # ACC5
        acc5 = accuracy_score(y_true_5, y_pred_5)
        # ACC2
        y_true_2 = [self.label_to_acc2(l) for l in y_true_5]
        y_pred_2 = [self.label_to_acc2(l) for l in y_pred_5]
        acc2 = accuracy_score(y_true_2, y_pred_2)
        # F1 (macro, 5类)
        f1 = f1_score(y_true_5, y_pred_5, average='macro', labels=['负','弱负','中','弱正','正'])


        self.plot_discrete_histogram(y_pred, y_true)
        
        return {'ACC5': acc5, 'ACC2': acc2, 'F1': f1}
    
    def plot_discrete_histogram(self, y_pred, y_true):
        # 统计每个离散值在预测值和真实标签中出现的次数
        import matplotlib.pyplot as plt

        discrete_values = self.DISCRETE_VALUES
        from collections import Counter

        # 统计频率
        pred_counts = Counter(y_pred)
        true_counts = Counter(y_true)

        # 构造完整的频率数组
        x = np.array(discrete_values)
        pred_freq = np.array([pred_counts.get(val, 0) for val in x])
        true_freq = np.array([true_counts.get(val, 0) for val in x])

        # 柱宽与位置
        bar_width = 0.08
        x_indices = np.arange(len(x))

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.bar(x_indices - bar_width/2, true_freq, width=bar_width, label='True Labels', color='orange')
        plt.bar(x_indices + bar_width/2, pred_freq, width=bar_width, label='Predicted Values', color='blue')

        # 设置刻度和标签
        plt.xticks(ticks=x_indices, labels=[f'{v:.1f}' for v in x])
        plt.xlabel('Discrete Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Discretized Predicted Values and True Labels')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()