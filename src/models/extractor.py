from torchvision.models import resnet50
import torch.nn as nn

class resnet50_extractor(nn.Module):
    def __init__(self, embedding_dim = 512, num_classes = 751):
        super().__init__()
        base_model = resnet50(weights="IMAGENET1K_V1")

        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,

            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Linear(2048, embedding_dim)

        self.dropout = nn.Dropout(0.5)

        self.bn = nn.BatchNorm1d(embedding_dim)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        feat = self.embedding(x)

        feat = self.bn(feat)

        feat = self.dropout(feat)

        emb = nn.functional.normalize(feat, p = 2,dim=1)

        logits = self.classifier(feat)

        return emb, logits
