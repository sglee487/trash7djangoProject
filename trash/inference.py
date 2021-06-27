import torch
import torch.nn as nn
import os
from torchvision import transforms

DATASET_NORMALIZE_INFO = {
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}

def simple_augment_test(
    dataset: str = "TACO", img_size: float = 224
) -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )

def inference(img):
    m = nn.Softmax(dim=1)
    BASE = os.path.dirname(os.path.abspath(__file__))
    pth_dir = os.path.join(BASE, "expsaves/mobilenetv3_7.pth")
    model3_7 = torch.load(pth_dir)
    model3_7.eval()
    img = torch.unsqueeze(simple_augment_test()(img).type(torch.FloatTensor), dim=0)
    result = model3_7(img)
    result_softmax = m(result)
    values, indexes = torch.topk(result_softmax, k=3, dim=1)
    values = torch.squeeze(values).tolist()
    indexes = torch.squeeze(indexes).tolist()
    return values, indexes