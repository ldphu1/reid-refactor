import torch
import numpy as np
from tqdm import tqdm
import sys

def evaluate_rankk(model, query_loader, gallery_loader, device, k=5):
    model.eval()
    query_embs, query_ids, query_cams = [], [], []
    gallery_embs, gallery_ids, gallery_cams = [], [], []

    with torch.no_grad():
        for img, pid, camid in tqdm(query_loader, desc="Extracting Query", leave=False, file=sys.stdout):
            emb, _ = model(img.to(device))
            query_embs.append(emb)
            query_ids.extend(pid)
            query_cams.extend(camid)

        for img, pid, camid in tqdm(gallery_loader, desc="Extracting Gallery", leave=False, file=sys.stdout):
            emb, _ = model(img.to(device))
            gallery_embs.append(emb)
            gallery_ids.extend(pid)
            gallery_cams.extend(camid)

    query_embs = torch.cat(query_embs, dim=0)
    gallery_embs = torch.cat(gallery_embs, dim=0)

    # convert to tensor
    query_ids = torch.tensor([int(x) for x in query_ids])
    gallery_ids = torch.tensor([int(x) for x in gallery_ids])

    query_cams = torch.tensor([int(x) for x in query_cams])
    gallery_cams = torch.tensor([int(x) for x in gallery_cams])

    # cosine similarity
    sims = torch.matmul(query_embs, gallery_embs.T)

    correct = 0
    for i in range(len(query_ids)):
        score = sims[i]

        keep_mask = ~((gallery_ids == query_ids[i]) & (gallery_cams == query_cams[i]))

        filtered_scores = score[keep_mask]
        filtered_gallery_ids = gallery_ids[keep_mask]

        if len(filtered_gallery_ids) == 0:
            continue

        actual_k = min(k, len(filtered_gallery_ids))

        _, indices = torch.topk(filtered_scores, k=actual_k, largest=True)
        indices = indices.cpu()

        if query_ids[i] in filtered_gallery_ids[indices]:
            correct += 1

    return correct / len(query_ids)

def evaluate_map(model, query_loader, gallery_loader, device):
    model.eval()

    query_embs, query_ids, query_cams = [], [], []
    gallery_embs, gallery_ids, gallery_cams = [], [], []

    with torch.no_grad():

        for img, pid, camid in query_loader:
            emb, _ = model(img.to(device))
            query_embs.append(emb)
            query_ids.extend(pid)
            query_cams.extend(camid)

        for img, pid, camid in gallery_loader:
            emb, _ = model(img.to(device))
            gallery_embs.append(emb)
            gallery_ids.extend(pid)
            gallery_cams.extend(camid)

    query_embs = torch.cat(query_embs)
    gallery_embs = torch.cat(gallery_embs)

    gallery_ids = np.array(gallery_ids)
    query_ids = np.array(query_ids)

    gallery_cams = np.array(gallery_cams)
    query_cams = np.array(query_cams)

    # cosine similarity
    sims = torch.matmul(query_embs, gallery_embs.T).cpu().numpy()

    APs = []

    for i in range(len(query_ids)):

        score = sims[i]

        keep_mask = ~((gallery_ids == query_ids[i]) & (gallery_cams == query_cams[i]))
        keep_mask = keep_mask & (gallery_ids != -1)

        filtered_scores = score[keep_mask]
        filtered_gallery_ids = gallery_ids[keep_mask]

        matches = (filtered_gallery_ids == query_ids[i]).astype(int)

        if matches.sum() == 0:
            continue

        order = np.argsort(-filtered_scores)
        matches_sorted = matches[order]
        cum_matches = np.cumsum(matches_sorted)
        precision = cum_matches / (np.arange(len(matches_sorted))+1)

        AP = (precision * matches_sorted).sum() / matches_sorted.sum()

        APs.append(AP)

    return np.mean(APs)