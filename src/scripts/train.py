import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from models import model
from dataset import *
from loss import *
from metrics import *

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

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    train_transform, val_transform = get_transforms()

    train_data = Market1501(root=os.path.join(args.data_root, "bounding_box_train"), transform=train_transform)
    query_data = Evaldataset(root=os.path.join(args.data_root, "query"), transform=val_transform)
    gallery_data = Evaldataset(root=os.path.join(args.data_root, "bounding_box_test"), transform=val_transform)

    sampler = MPerClassSampler(train_data.labels_list, m=4, batch_size=args.batch_size, length_before_new_iter=len(train_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True)

    query_loader = DataLoader(query_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    extractor = model.resnet50_extractor(embedding_dim=512).to(device)
    optimizer = Adam(extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss = BatchHardTripletLoss(margin=args.margin)

    global_step = 0
    best_rank1 = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, global_step = train_epoch(extractor, train_loader, optimizer, cross_entropy, triplet_loss, device, epoch, writer,
                                            global_step)
        print(f"Epoch [{epoch}/{args.epochs}] - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            print("Evaluating...")
            rank1 = evaluate_rankk(extractor, query_loader, gallery_loader, device, k=1)
            rank5 = evaluate_rankk(extractor, query_loader, gallery_loader, device, k=5)

            mAP = evaluate_map(extractor, query_loader, gallery_loader, device)

            writer.add_scalar("Metric/Rank1", rank1, epoch)
            writer.add_scalar("Metric/Rank5", rank5, epoch)
            writer.add_scalar("Metric/mAP", mAP, epoch)

            print(f"Rank-1: {rank1:.4f} | Rank-5: {rank5:.4f} | mAP: {mAP:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rank1': best_rank1
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"))

            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(extractor.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                print(f"Saved new best model with Rank-1: {best_rank1:.4f}")

        scheduler.step()

    torch.save(extractor.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
    writer.close()
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Re-ID Training Routine")

    parser.add_argument("--data_root", type=str, default="market1501",help="Path to Market1501 dataset")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save models")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.3, help="Margin for Triplet Loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="For CrossEntropy Loss")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for LR Scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for LR Scheduler")
    parser.add_argument("--eval_freq", type=int, default=5, help="Evaluate every N epochs")

    args = parser.parse_args()
    main(args)