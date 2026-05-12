import sys
import os
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import yaml
from src.models.extractor import resnet50_extractor
from src.dataset import *
from src.loss import *
from src.metrics import *


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_epoch(model, dataloader, optimizer, cross_entropy, triplet_loss, device, epoch, writer, global_step):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout)
    for imgs, labels in progress_bar:

        imgs = imgs.to(device)
        labels = labels.to(device)

        emb, logits = model(imgs)

        ce_loss = cross_entropy(logits, labels)
        trip_loss = triplet_loss(emb, labels)
        loss = trip_loss + ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging to TensorBoard
        total_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader), global_step

def main():
    os.makedirs(cfg["save_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["log_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    train_data = Market1501(root=os.path.join(cfg["dataset"]["data_root"], "bounding_box_train"), transform=train_transform)
    query_data = Evaldataset(root=os.path.join(cfg["dataset"]["data_root"], "query"), transform=val_transform)
    gallery_data = Evaldataset(root=os.path.join(cfg["dataset"]["data_root"], "bounding_box_test"), transform=val_transform)

    sampler = MPerClassSampler(train_data.labels_list, m=4, batch_size=cfg["dataset"]["batch_size"], length_before_new_iter=len(train_data))

    train_loader = DataLoader(train_data, batch_size=cfg["dataset"]["batch_size"], sampler=sampler, num_workers=cfg["dataset"]["num_worker"], drop_last=True)

    query_loader = DataLoader(query_data, batch_size=cfg["dataset"]["batch_size"], shuffle=False, num_workers=cfg["dataset"]["num_worker"])
    gallery_loader = DataLoader(gallery_data, batch_size=cfg["dataset"]["batch_size"], shuffle=False, num_workers=cfg["dataset"]["num_worker"])

    model = resnet50_extractor(embedding_dim=cfg["model"]["embedding_dim"], num_classes=cfg["model"]["num_classes"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"]["step_size"], gamma=cfg["train"]["gamma"])

    cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    triplet_loss = BatchHardTripletLoss(margin=cfg["train"]["margin"])

    global_step = 0
    best_rank1 = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        avg_loss, global_step = train_epoch(model, train_loader, optimizer, cross_entropy, triplet_loss, device, epoch, writer,
                                            global_step)
        print(f"Epoch [{epoch}/{cfg["train"]["epochs"]}] - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % cfg["train"]["eval_freq"] == 0 or epoch == cfg["train"]["epochs"]:
            print("Evaluating...")
            rank1 = evaluate_rankk(model, query_loader, gallery_loader, device, k=1)
            rank5 = evaluate_rankk(model, query_loader, gallery_loader, device, k=5)

            mAP = evaluate_map(model, query_loader, gallery_loader, device)

            writer.add_scalar("Metric/Rank1", rank1, epoch)
            writer.add_scalar("Metric/Rank5", rank5, epoch)
            writer.add_scalar("Metric/mAP", mAP, epoch)

            print(f"Rank-1: {rank1:.4f} | Rank-5: {rank5:.4f} | mAP: {mAP:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rank1': best_rank1
            }
            torch.save(checkpoint, os.path.join(cfg["save_dir"], f"model_epoch_{epoch}.pth"))

            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best_model.pth"))
                print(f"Saved new best model with Rank-1: {best_rank1:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "last_model.pth"))
    writer.close()
    print("DONE")

def log_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Re-ID Training Routine")
    parser.add_argument("--config", type=str, default=r"configs/train_config.yaml")
    args = parser.parse_args()

    cfg = log_config(args.config)

    main()