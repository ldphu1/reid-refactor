import argparse
import cv2
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO
from models import model

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_embedding(model, img_array, transform, device):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb, _ = model(img_tensor)

    return emb

track_identities = {}


def draw_fancy_bbox(frame, box, label, score, color):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"ID: {label} | {score:.2f}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if y1 - text_h - 10 < 0:
        back_y1, back_y2 = y1, y1 + text_h + 10
        text_y = y1 + text_h + 5
    else:
        back_y1, back_y2 = y1 - text_h - 10, y1
        text_y = y1 - 7

    cv2.rectangle(frame, (x1, back_y1), (x1 + text_w + 10, back_y2), color, -1)

    cv2.putText(frame, text, (x1 + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def process_video(args, reid_model, detector, gallery_data, transform, device):
    gallery_embs = gallery_data["embs"].to(device)
    gallery_ids = gallery_data["ids"]

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {args.video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    while True:
        flag, frame = cap.read()
        if not flag:
            break

        results = detector.track(frame, tracker = "bytetrack.yaml",persist=True, verbose=False, classes = [0])

        if results[0].boxes.id is None:
            out.write(frame)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 - x1 < 20 or y2 - y1 < 50:
                continue

            crop = frame[y1:y2, x1:x2]
            new_emb = extract_embedding(reid_model, crop, transform, device)

            if track_id not in track_identities:
                avg_emb = new_emb
            else:
                old_avg_emb = track_identities[track_id]["avg_emb"]
                avg_emb = args.alpha * new_emb + (1.0 - args.alpha) * old_avg_emb
                avg_emb = F.normalize(avg_emb, p=2, dim=1)

            sims = torch.matmul(avg_emb, gallery_embs.T)
            best_idx = torch.argmax(sims, dim=1).item()
            score = sims[0, best_idx].item()

            if score > args.threshold:
                pid = gallery_ids[best_idx]
                label_infor = {"PID": pid, "score": score, "color": (0, 255, 0), "avg_emb": avg_emb}

            draw_fancy_bbox(frame, (x1, y1, x2, y2), label_infor["PID"], label_infor["score"], label_infor["color"])

            track_identities[track_id] = label_infor

        out.write(frame)

        cv2.imshow("Re-ID Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="../data/video.mp4", help="Path to the original video")
    parser.add_argument("--output_path", type=str, default="output_reid.mp4", help="Path to save the resulting video")
    parser.add_argument("--model_weights", type=str, default="../weights/best_model.pth", help="Model weight file")
    parser.add_argument("--gallery_path", type=str, default="../weights/gallery_market1501.pt", help="Path to gallery file")
    parser.add_argument("--threshold", type=float, default=0.6, help="threshold")
    parser.add_argument("--alpha" ,type=float, default=0.1, help="Update rate for the moving average embedding. ")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    reid_model = model.resnet50_extractor(512)
    reid_model.load_state_dict(torch.load(args.model_weights, map_location=device))
    reid_model.to(device)
    reid_model.eval()

    gallery_data = torch.load(args.gallery_path)

    detector = YOLO("yolov8n.pt").to(device)

    transform = get_transform()

    process_video(args, reid_model, detector, gallery_data, transform, device)
    print("DONE")
