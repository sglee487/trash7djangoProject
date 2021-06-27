import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

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


with torch.no_grad():
    sample_img = np.random.rand(224,224,3)
    sample_img = torch.unsqueeze(simple_augment_test()(sample_img).type(torch.FloatTensor), dim=0)
    m = nn.Softmax(dim=1)
    m.eval()
    modelshuffle = torch.load("expsaves/shufflenetv205_238.pth")
    modelshuffle.eval()
    result1 = modelshuffle(sample_img)
    result1_softmax = m(result1)
    values1, indexes1 = torch.topk(result1_softmax, k=3, dim=1)
    values1 = torch.squeeze(values1)
    indexes1 = torch.squeeze(indexes1)
    values1 = values1.tolist()
    indexes1 = indexes1.tolist()

    model3_7 = torch.load("expsaves/mobilenetv3_7.pth")
    model3_7.eval()
    result2 = model3_7(sample_img)
    result2_softmax = m(result2)
    values2, indexes2 = torch.topk(result2_softmax, k=3, dim=1)
    values2 = torch.squeeze(values2)
    indexes2 = torch.squeeze(indexes2)
    values2 = values2.tolist()
    indexes2 = indexes2.tolist()

    model3_7_de04 = torch.load("expsaves/mobilenetv3_7_de04.pth")
    model3_7_de04.eval()
    result3 = model3_7_de04(sample_img)
    result3_softmax = m(result3)
    values3, indexes3 = torch.topk(result3_softmax, k=3, dim=1)
    values3 = torch.squeeze(values3)
    indexes3 = torch.squeeze(indexes3)
    values3 = values3.tolist()
    indexes3 = indexes3.tolist()