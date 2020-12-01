import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from pdf2image import convert_from_path

from . import detection


PIC_SIZE = 150
PROBABILITY_THRESHOLD = 0.8


# Model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvModel(nn.Module):
    def __init__(self, cat_n):
        super(ConvModel, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3),
                                 nn.MaxPool2d(2),
                                 nn.BatchNorm2d(out_ch))
        
        base = 10
        # (N, 1, 150, 150) -> (N, 512)
        self.conv = nn.Sequential(conv_block(1, base*3),
                                  conv_block(base*3, base*2),
                                  conv_block(base*2, base*3),
                                  conv_block(base*3, base*2),
                                  conv_block(base*2, base*3),
                                  Flatten())
        
        def linear_block(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(0.3),
                                 nn.Dropout(0.05))
        
        self.fc = nn.Sequential(linear_block(base*3*4, 256),
                                linear_block(256, cat_n))
        

    def forward(self, batch):
        
        def conv_pics(pic_list):
            return list(map(self.conv, list(map(lambda t: torch.unsqueeze(t, dim=0), pic_list))))
        
        conved = list(map(conv_pics, batch))
        conved = list(map(lambda lt: torch.stack(lt).sum(dim=0), conved))
        conved = torch.stack(conved)
        
        clss = self.fc(conved).squeeze(dim=1)
        
        return torch.sigmoid(clss)
    
    
def pilpaper2operations(pil_img):
    cv_img = detection.pil2cv(pil_img)
    cv_imgs = detection.crop_conturs(cv_img)
    pil_imgs = list(map(detection.cv2pil, cv_imgs))
    tensor_imgs = list(map(transform, pil_imgs))
    pred = model([tensor_imgs])
    probs = pred[0, [ixs]].detach().numpy()
    detail_ops = confident_ops[probs[0] > PROBABILITY_THRESHOLD]
    return list(detail_ops)


# all operations
all_ops = np.array(pickle.load(open('./predict_operations/all_ops.pkl', 'rb')))
# Confident predictable operations
ixs = [3, 6, 10, 12, 24, 34, 35, 36, 37, 39, 40, 41, 43, 44, 48, 54]
confident_ops = all_ops[ixs]


model = ConvModel(len(all_ops)).eval()
model.load_state_dict(torch.load('./predict_operations/conv_model.pt', map_location='cpu'))
transform = transforms.Compose([transforms.Grayscale(),
                             transforms.Resize((PIC_SIZE, PIC_SIZE)),
                             transforms.ToTensor()])