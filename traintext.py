from utils.SIMSData import SIMSLoader
from models.TextExtractor import TextExtractor
import torch
from torchinfo import summary

if __name__ == "__main__":
    dataloader = SIMSLoader(
        root="./data/ch-sims2s/ch-simsv2s", mode="text", batch_size=4, num_workers=4
    )
    
    train_loader = dataloader.trainloader
    test_loader = dataloader.testloader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = TextExtractor(
        pretrained_model=None,
        details=True,
        device=device,
        epochs=50,
        lr=2e-5,
        model_name="bert-base-chinese",
        save_path="./checkpoints/bert_sentiment_regressor.pth",
    )
    summary(
        extractor.model,
        input_data={
            "input_ids": torch.randint(0, 1000, (1, 64)).to(device),
            "token_type_ids": torch.zeros(1, 64, dtype=torch.int64).to(device),
            "attention_mask": torch.ones(1, 64).to(device),
        },
    )
    extractor.train(train_loader, test_loader)