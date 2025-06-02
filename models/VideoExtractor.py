from utils.load_data import SIMSData
import os
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Optional
from facenet_pytorch import MTCNN
import numpy as np
import json


# =============================
# 1. 情感分数 → 类别 映射函数
# =============================
def score_to_label(score: float) -> int:
    """
    把连续情感分数映射到 5 个类别：
      - 类别 0 (负):    -1.0, -0.8
      - 类别 1 (弱负):  -0.6, -0.4, -0.2
      - 类别 2 (中性):  0.0
      - 类别 3 (弱正):   0.2, 0.4, 0.6
      - 类别 4 (正):    0.8, 1.0
    """
    if score in (-1.0, -0.8):
        return '负'
    elif score in (-0.6, -0.4, -0.2):
        return '弱负'
    elif score == 0.0:
        return '中'
    elif score in (0.2, 0.4, 0.6):
        return '弱正'
    elif score in (0.8, 1.0):
        return '正'
    else:
        raise ValueError(f"Unexpected sentiment score: {score}")
    
def score_to_closest_value(score: float) -> float:
    """
    找到离输入分数最近的基准值：
    -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    如果输入分数正好在两个基准值中间（如 -0.7），则返回较大的值（如 -0.6）。
    """
    # 定义基准值列表（必须是升序排列）
    DISCRETE_VALUES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    values = np.array(DISCRETE_VALUES)
    return float(values[np.argmin(np.abs(values - score))])
    
# =============================
# 2. 视频情感回归分析器
# =============================

class VideoSentimentAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()  
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

    def _get_transform(self):
        """获取图像预处理管道"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def analyze_video(self, 
                      video_path: str, 
                      output_dir: Optional[str] = None, 
                      save_frames: bool = False,  # 重命名为save_frames
                      frame_interval: int = 10) -> Dict:
        """分析视频情感并输出连续分数"""
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频基础信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"视频信息: FPS={fps:.2f}, 时长={duration:.2f}秒")
        
        results = []
        processed_frames = []
        transform = self._get_transform()  # 图像预处理管道

        for frame_idx in tqdm(range(0, frame_count, frame_interval), desc="分析中"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: 
                break
                
            # 直接处理整帧图像，不需要人脸检测
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            image_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # 回归预测
            with torch.no_grad():
                pred_score = self.emotion_model(image_tensor).squeeze().item()
            predicted_label = score_to_label(score_to_closest_value(float(pred_score)))

            # 记录结果
            time_sec = frame_idx / fps
            results.append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "predicted_score": float(pred_score),
                "predicted_label": predicted_label
            })
            
            # 保存处理后的帧
            if save_frames and output_dir:
                # 在帧上绘制情感分数和标签
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, 
                           f"Score: {pred_score:.2f} ({predicted_label})",
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2)
                
                frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame_with_text)
                processed_frames.append({"frame_idx": frame_idx, "frame_path": frame_path})
        
        cap.release()
        return self._generate_report(video_path, fps, frame_count, results, output_dir, processed_frames, frame_interval)

    def _generate_report(self, video_path, fps, frame_count, results, output_dir, face_frames,frame_interval):
        """生成回归分析报告"""
        if not results:
            return self._get_empty_report(video_path, fps, frame_count)
        
        # 统计指标计算
        scores = np.array([r["predicted_score"] for r in results])
        report = {
            "video_path": video_path,
            "fps": float(fps),
            "total_frames": frame_count,
            "analyzed_frames": len(results),
            "face_detection_rate": len(results) / max(1, frame_count//frame_interval),
            "prediction_stats": {
                "average_score": float(np.mean(scores)),
                "overall_label": score_to_label(score_to_closest_value(float(np.mean(scores)))),
                "std_dev": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores))
            },
            "frame_details": results,
        }
        # 保存结果到文件
        if output_dir:
            self._save_results(report, output_dir)
        return report

    def _get_empty_report(self, video_path, fps, frame_count):
        """未检测到人脸时的空报告"""
        return {
            "video_path": video_path,
            "fps": float(fps),
            "total_frames": frame_count,
            "analyzed_frames": 0,
            "face_detection_rate": 0,
            "prediction_stats": {},
            "frame_details": [],
        }
    

    def _save_results(self, report, output_dir):
        """保存分析结果到JSON和CSV"""
        # 保存JSON报告
        json_path = os.path.join(output_dir, "regression_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_types(report), f, ensure_ascii=False, indent=2)
        
        # 保存CSV细节
        df = pd.DataFrame(report["frame_details"])
        csv_path = os.path.join(output_dir, "frame_details.csv")
        df.to_csv(csv_path, index=False)
        print(f"结果已保存至: {output_dir}")

    @staticmethod
    def _convert_types(data):
        """转换非JSON兼容的数据类型"""
        if isinstance(data, (np.int64, np.int32)): return int(data)
        if isinstance(data, (np.float32, np.float64)): return float(data)
        if isinstance(data, dict): return {k: VideoSentimentAnalyzer._convert_types(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)): return [VideoSentimentAnalyzer._convert_types(item) for item in data]
        return data


class VideoSentimentDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 mode: str, 
                 analyzer: VideoSentimentAnalyzer, 
                 frame_interval: int = 10):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录
            mode: 数据模式 (train/val/test)
            analyzer: 情感分析器实例
            frame_interval: 帧间隔
        """
        self.analyzer = analyzer
        self.frame_interval = frame_interval
        self.mode = mode
        
        # 使用SIMSData获取元数据并添加路径列
        self.meta_df = self._get_meta(root_dir, mode)

    def _get_meta(self, root_dir: str, mode: str) -> pd.DataFrame:
        """获取并处理元数据，添加视频文件路径列"""
        # 使用SIMSData获取原始元数据
        meta = SIMSData(root=root_dir, mode=mode).meta
        
        # 添加完整视频路径列
        meta['path'] = meta.apply(
            lambda row: os.path.join(root_dir, "Raw", row['video_id'], f"{row['clip_id']}.mp4"),
            axis=1
        )
        
        return meta.reset_index(drop=True)

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        """
        返回第 idx 条视频对应的 (frames_list, target_score)
        frames_list: List[Tensor], shape (3,224,224) 每个元素是一帧经过 transform 后的张量
        target_score: float, 从 meta_df 对应的列读取
        """
        row = self.meta_df.iloc[idx]
        video_path = row['path']
        # 假设元数据里有一列叫 "score"：如果实际列名不同，请自行替换
        target_score = float(row['label_V'])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frames_tensors: List[torch.Tensor] = []
        transform = self.analyzer._get_transform()  # 调用你之前定义的那个 transform
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 逐帧抽样
        for frame_idx in range(0, frame_count, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)  # 形状 (3,224,224)
            frames_tensors.append(tensor)

        cap.release()
        # 如果一段视频抽不到任何帧，则 frames_tensors 可能为空——这里简单跳过：
        if len(frames_tensors) == 0:
            # 为了不破坏 batch，大多数做法是至少返回一帧全零张量
            fake = torch.zeros(3, 224, 224)
            frames_tensors = [fake]

        return frames_tensors, target_score

def train_and_validate(
    root_dir: str,
    device: torch.device,
    batch_size: int = 4,
    num_epochs: int = 10,
    frame_interval: int = 10,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "./checkpoints"
):
    """
    root_dir: SIMS 数据集根目录
    device: torch.device("cuda") / torch.device("cpu")
    batch_size: 每个 batch 里包含多少个视频
    num_epochs: 训练多少个 epoch
    frame_interval: 抽帧间隔
    learning_rate: 学习率
    checkpoint_dir: 用来保存最优模型参数
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ——1. 初始化 分析器 + 数据集 + DataLoader —— #
    analyzer = VideoSentimentAnalyzer()
    analyzer.to(device)
    analyzer.device = device

    # train 集
    train_dataset = VideoSentimentDataset(
        root_dir=root_dir,
        mode="train",
        analyzer=analyzer,
        frame_interval=frame_interval
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch  # 让一个 batch 返回 List[(frames_list, target_score), ...]
    )

    # valid 集
    valid_dataset = VideoSentimentDataset(
        root_dir=root_dir,
        mode="valid",
        analyzer=analyzer,
        frame_interval=frame_interval
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch
    )

    # ——2. 定义优化器 + 损失函数 —— #
    optimizer = torch.optim.Adam(analyzer.backbone.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        analyzer.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in pbar:
            # batch 是一个 List，长度 = batch_size，每个元素是 (frames_list, target_score)
            optimizer.zero_grad()
            batch_loss = 0.0

            for frames_list, target_score in batch:
                # frames_list: List[Tensor]，把它们拼成一个大的张量方便并行
                # e.g. frames_tensor_batch.shape = (N_frames, 3, 224, 224)
                frames_tensor = torch.stack(frames_list, dim=0).to(device)  # N_frames × 3 × 224 ×224
                target = torch.tensor([target_score], dtype=torch.float32, device=device)

                # 对同一段视频的每一帧都做一次前向
                preds = analyzer(frames_tensor)  # preds.shape = (N_frames, 1)
                preds = preds.squeeze(1)  # 变成 (N_frames,)

                # 取这段视频中所有帧预测分的平均作为视频级预测
                video_pred = preds.mean().unsqueeze(0)  # (1,)
                loss = criterion(video_pred, target)
                batch_loss += loss

            # 一次 batch 内所有视频的 loss 累加后再做 backward
            batch_loss = batch_loss / len(batch)  # 也可以选择不平均
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            n_batches += 1
            pbar.set_postfix(train_loss=f"{running_loss / n_batches:.4f}")

        avg_train_loss = running_loss / max(1, n_batches)

        # ——3. 在验证集上评估 —— #
        analyzer.eval()
        val_running_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            pbar_val = tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]")
            for batch in pbar_val:
                batch_val_loss = 0.0
                for frames_list, target_score in batch:
                    frames_tensor = torch.stack(frames_list, dim=0).to(device)
                    target = torch.tensor([target_score], dtype=torch.float32, device=device)

                    preds = analyzer(frames_tensor).squeeze(1)  # (N_frames,)
                    video_pred = preds.mean().unsqueeze(0)  # (1,)
                    loss = criterion(video_pred, target)
                    batch_val_loss += loss

                batch_val_loss = batch_val_loss / len(batch)
                val_running_loss += batch_val_loss.item()
                val_n_batches += 1
                pbar_val.set_postfix(valid_loss=f"{val_running_loss / val_n_batches:.4f}")

        avg_val_loss = val_running_loss / max(1, val_n_batches)
        print(f"Epoch {epoch} ► Train Loss: {avg_train_loss:.4f} │ Val Loss: {avg_val_loss:.4f}")

        # 如果当前验证 Loss 更好，则保存模型
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}.pth")
            torch.save(analyzer.state_dict(), checkpoint_path)
            print(f"【Saved Best Model】 Epoch {epoch}, Val Loss: {avg_val_loss:.4f}")

    print(f"训练结束。最佳模型出现在 Epoch {best_epoch}, 验证 Loss={best_valid_loss:.4f}")
    return analyzer, best_epoch, best_valid_loss


# =============================================================================
# 3. 测试流程：用训练好的权重，对 test 集逐条视频调用 analyze_video，自动生成 report
# =============================================================================
def test_and_save_reports(
    root_dir: str,
    checkpoint_path: str,
    device: torch.device,
    frame_interval: int = 10,
    output_root: str = "./test_reports"
):
    """
    root_dir: 数据集根目录
    checkpoint_path: 训练好的模型参数文件 (.pth)
    device: torch.device("cuda") / torch.device("cpu")
    frame_interval: 抽帧间隔
    output_root: 所有测试报告的根目录。每个视频会对应一个子文件夹。
    """
    os.makedirs(output_root, exist_ok=True)

    # ——1. 重新构造 analyzer，并加载权重 —— #
    analyzer = VideoSentimentAnalyzer()
    analyzer.to(device)
    analyzer.emotion_model = analyzer
    analyzer.device = device

    state_dict = torch.load(checkpoint_path, map_location=device)
    analyzer.load_state_dict(state_dict)
    analyzer.eval()

    # ——2. 用 VideoSentimentDataset 列出所有 test 视频路径，但不抽帧；只要路径即可 —— #
    # 这里直接用原始 VideoSentimentDataset（它没有 __getitem__），只为拿到 meta_df
    temp_ds = VideoSentimentDataset(root_dir=root_dir, mode="test", analyzer=analyzer, frame_interval=frame_interval)
    meta_df_test: pd.DataFrame = temp_ds.meta_df  # 包含每条 test 视频的 "video_id", "clip_id", "path" 等信息

    # 对 test 集的每一行视频，都调用 analyze_video，把报告写到独立的文件夹
    for idx, row in tqdm(meta_df_test.iterrows(), total=len(meta_df_test), desc="Testing"):
        video_id = row['video_id']
        clip_id = row['clip_id']
        video_path = row['path']

        # 给每条视频创建一个子文件夹，目录名可以是 “videoID_clipID”
        subdir = os.path.join(output_root, f"{video_id}_{clip_id}")
        os.makedirs(subdir, exist_ok=True)

        # analyze_video 会自动把 JSON + CSV 写到 subdir
        try:
            report = analyzer.analyze_video(
                video_path=video_path,
                output_dir=subdir,
                save_frames=False,        # 不保存带文字的帧，如果想也可以置 True
                frame_interval=frame_interval
            )
            # report 已经包含了所有统计信息，如果你想额外保存完整 report 对象，也可以：
            # with open(os.path.join(subdir, "full_report.pkl"), "wb") as f:
            #     pickle.dump(report, f)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print(f"测试结束。所有报告保存在：{output_root}")


# =============================================================================
# 4. 脚本入口（示例用法）
# =============================================================================
if __name__ == "__main__":
    # 确保在你本机/服务器上选一个可用的 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_and_validate(
            root_dir="data\ch-sims2s\ch-simsv2s",
            device=device,
            batch_size=32,
            num_epochs=20,
            frame_interval=10,
            learning_rate=1e-4,
            checkpoint_dir="checkpoint"
        )
from utils.load_data import SIMSData
import os
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Optional
from facenet_pytorch import MTCNN
import numpy as np
import json


# =============================
# 1. 情感分数 → 类别 映射函数
# =============================
def score_to_label(score: float) -> int:
    """
    把连续情感分数映射到 5 个类别：
      - 类别 0 (负):    -1.0, -0.8
      - 类别 1 (弱负):  -0.6, -0.4, -0.2
      - 类别 2 (中性):  0.0
      - 类别 3 (弱正):   0.2, 0.4, 0.6
      - 类别 4 (正):    0.8, 1.0
    """
    if score in (-1.0, -0.8):
        return '负'
    elif score in (-0.6, -0.4, -0.2):
        return '弱负'
    elif score == 0.0:
        return '中'
    elif score in (0.2, 0.4, 0.6):
        return '弱正'
    elif score in (0.8, 1.0):
        return '正'
    else:
        raise ValueError(f"Unexpected sentiment score: {score}")
    
def score_to_closest_value(score: float) -> float:
    """
    找到离输入分数最近的基准值：
    -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    如果输入分数正好在两个基准值中间（如 -0.7），则返回较大的值（如 -0.6）。
    """
    # 定义基准值列表（必须是升序排列）
    DISCRETE_VALUES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    values = np.array(DISCRETE_VALUES)
    return float(values[np.argmin(np.abs(values - score))])
    
# =============================
# 2. 视频情感回归分析器
# =============================

class VideoSentimentAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()  
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

    def _get_transform(self):
        """获取图像预处理管道"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def analyze_video(self, 
                      video_path: str, 
                      output_dir: Optional[str] = None, 
                      save_frames: bool = False,  # 重命名为save_frames
                      frame_interval: int = 10) -> Dict:
        """分析视频情感并输出连续分数"""
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频基础信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"视频信息: FPS={fps:.2f}, 时长={duration:.2f}秒")
        
        results = []
        processed_frames = []
        transform = self._get_transform()  # 图像预处理管道

        for frame_idx in tqdm(range(0, frame_count, frame_interval), desc="分析中"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: 
                break
                
            # 直接处理整帧图像，不需要人脸检测
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            image_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # 回归预测
            with torch.no_grad():
                pred_score = self.emotion_model(image_tensor).squeeze().item()
            predicted_label = score_to_label(score_to_closest_value(float(pred_score)))

            # 记录结果
            time_sec = frame_idx / fps
            results.append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "predicted_score": float(pred_score),
                "predicted_label": predicted_label
            })
            
            # 保存处理后的帧
            if save_frames and output_dir:
                # 在帧上绘制情感分数和标签
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, 
                           f"Score: {pred_score:.2f} ({predicted_label})",
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2)
                
                frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame_with_text)
                processed_frames.append({"frame_idx": frame_idx, "frame_path": frame_path})
        
        cap.release()
        return self._generate_report(video_path, fps, frame_count, results, output_dir, processed_frames, frame_interval)

    def _generate_report(self, video_path, fps, frame_count, results, output_dir, face_frames,frame_interval):
        """生成回归分析报告"""
        if not results:
            return self._get_empty_report(video_path, fps, frame_count)
        
        # 统计指标计算
        scores = np.array([r["predicted_score"] for r in results])
        report = {
            "video_path": video_path,
            "fps": float(fps),
            "total_frames": frame_count,
            "analyzed_frames": len(results),
            "face_detection_rate": len(results) / max(1, frame_count//frame_interval),
            "prediction_stats": {
                "average_score": float(np.mean(scores)),
                "overall_label": score_to_label(score_to_closest_value(float(np.mean(scores)))),
                "std_dev": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores))
            },
            "frame_details": results,
        }
        # 保存结果到文件
        if output_dir:
            self._save_results(report, output_dir)
        return report

    def _get_empty_report(self, video_path, fps, frame_count):
        """未检测到人脸时的空报告"""
        return {
            "video_path": video_path,
            "fps": float(fps),
            "total_frames": frame_count,
            "analyzed_frames": 0,
            "face_detection_rate": 0,
            "prediction_stats": {},
            "frame_details": [],
        }
    

    def _save_results(self, report, output_dir):
        """保存分析结果到JSON和CSV"""
        # 保存JSON报告
        json_path = os.path.join(output_dir, "regression_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_types(report), f, ensure_ascii=False, indent=2)
        
        # 保存CSV细节
        df = pd.DataFrame(report["frame_details"])
        csv_path = os.path.join(output_dir, "frame_details.csv")
        df.to_csv(csv_path, index=False)
        print(f"结果已保存至: {output_dir}")

    @staticmethod
    def _convert_types(data):
        """转换非JSON兼容的数据类型"""
        if isinstance(data, (np.int64, np.int32)): return int(data)
        if isinstance(data, (np.float32, np.float64)): return float(data)
        if isinstance(data, dict): return {k: VideoSentimentAnalyzer._convert_types(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)): return [VideoSentimentAnalyzer._convert_types(item) for item in data]
        return data


class VideoSentimentDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 mode: str, 
                 analyzer: VideoSentimentAnalyzer, 
                 frame_interval: int = 10):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录
            mode: 数据模式 (train/val/test)
            analyzer: 情感分析器实例
            frame_interval: 帧间隔
        """
        self.analyzer = analyzer
        self.frame_interval = frame_interval
        self.mode = mode
        
        # 使用SIMSData获取元数据并添加路径列
        self.meta_df = self._get_meta(root_dir, mode)

    def _get_meta(self, root_dir: str, mode: str) -> pd.DataFrame:
        """获取并处理元数据，添加视频文件路径列"""
        # 使用SIMSData获取原始元数据
        meta = SIMSData(root=root_dir, mode=mode).meta
        
        # 添加完整视频路径列
        meta['path'] = meta.apply(
            lambda row: os.path.join(root_dir, "Raw", row['video_id'], f"{row['clip_id']}.mp4"),
            axis=1
        )
        
        return meta.reset_index(drop=True)

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        """
        返回第 idx 条视频对应的 (frames_list, target_score)
        frames_list: List[Tensor], shape (3,224,224) 每个元素是一帧经过 transform 后的张量
        target_score: float, 从 meta_df 对应的列读取
        """
        row = self.meta_df.iloc[idx]
        video_path = row['path']
        # 假设元数据里有一列叫 "score"：如果实际列名不同，请自行替换
        target_score = float(row['label_V'])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frames_tensors: List[torch.Tensor] = []
        transform = self.analyzer._get_transform()  # 调用你之前定义的那个 transform
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 逐帧抽样
        for frame_idx in range(0, frame_count, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)  # 形状 (3,224,224)
            frames_tensors.append(tensor)

        cap.release()
        # 如果一段视频抽不到任何帧，则 frames_tensors 可能为空——这里简单跳过：
        if len(frames_tensors) == 0:
            # 为了不破坏 batch，大多数做法是至少返回一帧全零张量
            fake = torch.zeros(3, 224, 224)
            frames_tensors = [fake]

        return frames_tensors, target_score

def train_and_validate(
    root_dir: str,
    device: torch.device,
    batch_size: int = 4,
    num_epochs: int = 10,
    frame_interval: int = 10,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "./checkpoints"
):
    """
    root_dir: SIMS 数据集根目录
    device: torch.device("cuda") / torch.device("cpu")
    batch_size: 每个 batch 里包含多少个视频
    num_epochs: 训练多少个 epoch
    frame_interval: 抽帧间隔
    learning_rate: 学习率
    checkpoint_dir: 用来保存最优模型参数
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ——1. 初始化 分析器 + 数据集 + DataLoader —— #
    analyzer = VideoSentimentAnalyzer()
    analyzer.to(device)
    analyzer.device = device

    # train 集
    train_dataset = VideoSentimentDataset(
        root_dir=root_dir,
        mode="train",
        analyzer=analyzer,
        frame_interval=frame_interval
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch  # 让一个 batch 返回 List[(frames_list, target_score), ...]
    )

    # valid 集
    valid_dataset = VideoSentimentDataset(
        root_dir=root_dir,
        mode="valid",
        analyzer=analyzer,
        frame_interval=frame_interval
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch
    )

    # ——2. 定义优化器 + 损失函数 —— #
    optimizer = torch.optim.Adam(analyzer.backbone.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        analyzer.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in pbar:
            # batch 是一个 List，长度 = batch_size，每个元素是 (frames_list, target_score)
            optimizer.zero_grad()
            batch_loss = 0.0

            for frames_list, target_score in batch:
                # frames_list: List[Tensor]，把它们拼成一个大的张量方便并行
                # e.g. frames_tensor_batch.shape = (N_frames, 3, 224, 224)
                frames_tensor = torch.stack(frames_list, dim=0).to(device)  # N_frames × 3 × 224 ×224
                target = torch.tensor([target_score], dtype=torch.float32, device=device)

                # 对同一段视频的每一帧都做一次前向
                preds = analyzer(frames_tensor)  # preds.shape = (N_frames, 1)
                preds = preds.squeeze(1)  # 变成 (N_frames,)

                # 取这段视频中所有帧预测分的平均作为视频级预测
                video_pred = preds.mean().unsqueeze(0)  # (1,)
                loss = criterion(video_pred, target)
                batch_loss += loss

            # 一次 batch 内所有视频的 loss 累加后再做 backward
            batch_loss = batch_loss / len(batch)  # 也可以选择不平均
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            n_batches += 1
            pbar.set_postfix(train_loss=f"{running_loss / n_batches:.4f}")

        avg_train_loss = running_loss / max(1, n_batches)

        # ——3. 在验证集上评估 —— #
        analyzer.eval()
        val_running_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            pbar_val = tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]")
            for batch in pbar_val:
                batch_val_loss = 0.0
                for frames_list, target_score in batch:
                    frames_tensor = torch.stack(frames_list, dim=0).to(device)
                    target = torch.tensor([target_score], dtype=torch.float32, device=device)

                    preds = analyzer(frames_tensor).squeeze(1)  # (N_frames,)
                    video_pred = preds.mean().unsqueeze(0)  # (1,)
                    loss = criterion(video_pred, target)
                    batch_val_loss += loss

                batch_val_loss = batch_val_loss / len(batch)
                val_running_loss += batch_val_loss.item()
                val_n_batches += 1
                pbar_val.set_postfix(valid_loss=f"{val_running_loss / val_n_batches:.4f}")

        avg_val_loss = val_running_loss / max(1, val_n_batches)
        print(f"Epoch {epoch} ► Train Loss: {avg_train_loss:.4f} │ Val Loss: {avg_val_loss:.4f}")

        # 如果当前验证 Loss 更好，则保存模型
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}.pth")
            torch.save(analyzer.state_dict(), checkpoint_path)
            print(f"【Saved Best Model】 Epoch {epoch}, Val Loss: {avg_val_loss:.4f}")

    print(f"训练结束。最佳模型出现在 Epoch {best_epoch}, 验证 Loss={best_valid_loss:.4f}")
    return analyzer, best_epoch, best_valid_loss


# =============================================================================
# 3. 测试流程：用训练好的权重，对 test 集逐条视频调用 analyze_video，自动生成 report
# =============================================================================
def test_and_save_reports(
    root_dir: str,
    checkpoint_path: str,
    device: torch.device,
    frame_interval: int = 10,
    output_root: str = "./test_reports"
):
    """
    root_dir: 数据集根目录
    checkpoint_path: 训练好的模型参数文件 (.pth)
    device: torch.device("cuda") / torch.device("cpu")
    frame_interval: 抽帧间隔
    output_root: 所有测试报告的根目录。每个视频会对应一个子文件夹。
    """
    os.makedirs(output_root, exist_ok=True)

    # ——1. 重新构造 analyzer，并加载权重 —— #
    analyzer = VideoSentimentAnalyzer()
    analyzer.to(device)
    analyzer.emotion_model = analyzer
    analyzer.device = device

    state_dict = torch.load(checkpoint_path, map_location=device)
    analyzer.load_state_dict(state_dict)
    analyzer.eval()

    # ——2. 用 VideoSentimentDataset 列出所有 test 视频路径，但不抽帧；只要路径即可 —— #
    # 这里直接用原始 VideoSentimentDataset（它没有 __getitem__），只为拿到 meta_df
    temp_ds = VideoSentimentDataset(root_dir=root_dir, mode="test", analyzer=analyzer, frame_interval=frame_interval)
    meta_df_test: pd.DataFrame = temp_ds.meta_df  # 包含每条 test 视频的 "video_id", "clip_id", "path" 等信息

    # 对 test 集的每一行视频，都调用 analyze_video，把报告写到独立的文件夹
    for idx, row in tqdm(meta_df_test.iterrows(), total=len(meta_df_test), desc="Testing"):
        video_id = row['video_id']
        clip_id = row['clip_id']
        video_path = row['path']

        # 给每条视频创建一个子文件夹，目录名可以是 “videoID_clipID”
        subdir = os.path.join(output_root, f"{video_id}_{clip_id}")
        os.makedirs(subdir, exist_ok=True)

        # analyze_video 会自动把 JSON + CSV 写到 subdir
        try:
            report = analyzer.analyze_video(
                video_path=video_path,
                output_dir=subdir,
                save_frames=False,        # 不保存带文字的帧，如果想也可以置 True
                frame_interval=frame_interval
            )
            # report 已经包含了所有统计信息，如果你想额外保存完整 report 对象，也可以：
            # with open(os.path.join(subdir, "full_report.pkl"), "wb") as f:
            #     pickle.dump(report, f)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print(f"测试结束。所有报告保存在：{output_root}")


# =============================================================================
# 4. 脚本入口（示例用法）
# =============================================================================
if __name__ == "__main__":
    # 确保在你本机/服务器上选一个可用的 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_and_validate(
            root_dir="data\ch-sims2s\ch-simsv2s",
            device=device,
            batch_size=32,
            num_epochs=20,
            frame_interval=10,
            learning_rate=1e-4,
            checkpoint_dir="checkpoint"
        )
