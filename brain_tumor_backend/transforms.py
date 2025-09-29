
import torchvision.transforms as T
IM_SIZE = 224
IM_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(IM_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
