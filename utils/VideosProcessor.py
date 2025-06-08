import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


class VideosProcessor:
    """
    使用OpenFace处理视频并提取面部特征的工具类

    功能包括：
    - 批量处理单个视频或目录中的多个视频
    - 从OpenFace生成的CSV文件中提取并归一化面部特征
    - 将处理后的特征保存为pickle文件
    """

    def __init__(self, openface_exe_path):
        """
        初始化OpenFace视频处理器

        参数:
            openface_exe_path (str): FeatureExtraction.exe可执行文件的路径
            video_extensions (List[str], optional): 支持的视频格式列表，默认为常见格式
        """
        self.openface_exe = os.path.abspath(openface_exe_path)
        self.video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

        # 检查OpenFace可执行文件是否存在
        if not os.path.exists(self.openface_exe):
            raise FileNotFoundError(f"OpenFace可执行文件不存在: {self.openface_exe}")

    def process_videos(self, input_path: str, output_dir: str) -> int:
        """
        使用OpenFace处理视频文件（支持单个文件或目录）

        参数:
            input_path (str): 输入视频文件路径或目录路径
            output_dir (str): 输出结果目录路径

        返回:
            int: 成功处理的视频数量
        """
        # 标准化路径
        input_path = os.path.abspath(input_path)
        output_dir = os.path.abspath(output_dir)

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 检查输入是文件还是目录
        video_files = []
        if os.path.isfile(input_path):
            # 处理单个文件
            if Path(input_path).suffix.lower() in self.video_extensions:
                video_files = [input_path]
            else:
                raise ValueError(f"输入文件格式不支持: {Path(input_path).suffix}")
        elif os.path.isdir(input_path):
            # 处理目录中的所有视频
            video_files = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if os.path.isfile(os.path.join(input_path, f))
                and Path(f).suffix.lower() in self.video_extensions
            ]
        else:
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

        # 检查是否有视频文件
        if not video_files:
            print(f"警告: 未找到视频文件！")
            return 0

        print(f"找到 {len(video_files)} 个视频文件，开始处理...")

        # 循环处理每个视频文件
        success_count = 0
        for video_path in video_files:
            file_name = Path(video_path).stem  # 获取文件名（不含扩展名）

            print(f"\n正在处理: {file_name}")

            # 构建OpenFace命令
            command = [
                self.openface_exe,
                "-f",
                video_path,
                "-out_dir",
                output_dir,
                "-2Dfp",
                "-3Dfp",
                "-pose",
                "-aus",
                "-gaze",
                "-no2Dvid",
                "-no3Dvid",
                "-noMovid",
                "-nomask",
            ]

            try:
                # 执行命令并捕获输出
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                print(f"成功处理: {file_name}")
                success_count += 1

            except subprocess.CalledProcessError as e:
                print(f"处理失败: {file_name}")
                print(f"错误信息: {e.stderr}")
            except Exception as e:
                print(f"发生未知错误: {e}")

        print(
            f"\n处理完成！成功: {success_count}, 失败: {len(video_files) - success_count}"
        )
        return success_count

    def extract_and_normalize_features(
        self, csv_folder: str, output_pkl_path: str
    ) -> Dict[str, Any]:
        """
        提取面部特征（面部标志、AU、头部姿势、注视方向）并进行归一化，保存为.pkl文件。

        参数:
            csv_folder (str): 包含多个视频特征CSV文件的文件夹路径
            output_pkl_path (str): 输出.pkl文件保存路径

        返回:
            Dict[str, Any]: 包含处理后特征和元数据的字典
        """

        def extract_selected_features(df):
            """提取指定特征列（只保留成功检测的帧）"""
            # df = df[df[' success'] == 1]  # 只保留成功检测的帧

            prefixes = [
                # ' x_', ' y_', ' X_', ' Y_', ' Z_',     # 面部标志（2D/3D）
                " AU"  # 面部动作单位
                # ' pose_T', ' pose_R',              # 头部位置与旋转
                # ' gaze_'                           # 眼睛注视方向
            ]
            selected_cols = [
                col for col in df.columns if any(col.startswith(p) for p in prefixes)
            ]
            return df[selected_cols], selected_cols

        video_data = {}
        all_features = []
        column_names = None

        for filename in os.listdir(csv_folder):
            if filename.endswith(".csv"):
                video_id = os.path.splitext(filename)[0]
                path = os.path.join(csv_folder, filename)
                try:
                    df = pd.read_csv(path)
                    features_df, selected_cols = extract_selected_features(df)

                    if column_names is None:
                        column_names = selected_cols

                    features = features_df.values

                    if len(features) == 0:
                        print(f"No valid frames in {video_id}")
                        continue

                    video_data[video_id] = {"vision": features, "length": len(features)}
                    all_features.append(features)
                    print(f"Processed {video_id} ({len(features)} frames)")
                except Exception as e:
                    print(f"Error in {filename}: {e}")

        # 归一化
        scaler = StandardScaler()
        scaler.fit(np.vstack(all_features))

        # 对每个视频的特征进行归一化
        for vid in video_data:
            video_data[vid]["vision"] = scaler.transform(video_data[vid]["vision"])

        output = {
            "videos": video_data,
            "feature_names": column_names,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
        }

        with open(output_pkl_path, "wb") as f:
            pickle.dump(output, f)

        print(f"\n🎉 Saved normalized features to: {output_pkl_path}")
        print(f"📐 Final feature dimension: {len(column_names)}")

        return output


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = VideosProcessor(
        openface_exe_path=r"E:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    )

    # 处理单个视频
    # processor.process_videos(
    #     input_path=r"E:\Desk\msa\Multimodal_emotional_analysis\OpenFace_Video\MergedVideos\aqgy3_0001_00000.mp4",
    #     output_dir=r"E:\Desk\msa\Multimodal_emotional_analysis\OpenFace_Video\FeatureExtractedVideos"
    # )

    # 处理目录中的所有视频
    # processor.process_videos(
    #     input_path=r"E:\Desk\msa\Multimodal_emotional_analysis\OpenFace_Video\MergedVideos",
    #     output_dir=r"E:\Desk\msa\Multimodal_emotional_analysis\OpenFace_Video\FeatureExtractedVideos"
    # )

    # 处理特征保存为pkl文件
    csv_input_folder = r"E:\Desk\msa\Multimodal_emotional_analysis\OpenFace_Video\FeatureExtractedVideos"
    output_pkl_file = r"E:\Desk\msa\Multimodal_emotional_analysis\data\ch-sims2s\openface_features_AU.pkl"
    processor.extract_and_normalize_features(csv_input_folder, output_pkl_file)
