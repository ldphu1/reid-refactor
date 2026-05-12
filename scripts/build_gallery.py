import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from src.models.extractor import resnet50_extractor
import yaml

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_embedding(model, img_path, transform, device):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb, _ = model(img)

    return emb.cpu()


def build_gallery(data_dir, model_path, save_path, device):
    model = resnet50_extractor(512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform()

    gallery_embs = []
    gallery_ids = []
    gallery_paths = []

    valid_images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    for img_name in tqdm(valid_images):
        full_path = os.path.join(data_dir, img_name)

        emb = extract_embedding(model, full_path, transform, device)
        pid = img_name.split("_")[0]

        if pid in ['0000', '-1']:
            continue

        gallery_paths.append(full_path)
        gallery_embs.append(emb)
        gallery_ids.append(pid)

    gallery_embs = torch.cat(gallery_embs)

    print(f"Saving features to {save_path}...")
    torch.save({
        "embs": gallery_embs,
        "ids": gallery_ids,
        "paths": gallery_paths
    }, save_path)
    print("Done!")

def log_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and build gallery features for Re-ID")
    parser.add_argument("--config", type=str, default=r"src/configs/build_gallery_config.yaml")
    args = parser.parse_args()

    cfg = log_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    build_gallery(cfg["data_dir"], cfg["model_path"], cfg["save_path"], device)
