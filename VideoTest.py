from models.VideoExtractor import VideoExtractor

if __name__ == "__main__":
    modelname =  "conv_tf_AU_smoothloss_v3"
    # print("=== 训练模式示例 ===")
    # trainer = VideoExtractor(type="train", details=True,model_name = modelname)
    # print(trainer.DEVICE)
    # trainer.train()

    print("=== 测试模式示例 ===")
    tester = VideoExtractor(
        type="load",
        details=True,
        model_dir="saved_model",
        model_name=modelname
    )
    
    #运行测试集评价
    test_results = tester.test_model()
    # # 输出关键指标
    print(f"关键指标 - MSE: {test_results['mse']:.4f}, ACC5: {test_results['acc5']:.4f}")

    #print("=== 预测模式示例 ===")
    result1 = tester.predict(video_path = r"C:\Users\Cinyarn\Downloads\Demo_cry.mp4", openface_exe_path =r"E:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")
    print(result1)
    result2 = tester.predict(video_path = r"C:\Users\Cinyarn\Downloads\Demo_laugh.mp4", openface_exe_path =r"E:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")
    print(result2)