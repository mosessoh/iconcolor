import argparse
from PIL import Image
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, utils

################################################
# These functions were extracted from momo_utils.py as a quick fix to
# preventing users from having to download all the training dependencies

class CentraliseImage(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        size = (self.output_size,self.output_size)
        layer = Image.new
        layer = Image.new("RGB", size, (255,255,255))
        layer.paste(img, tuple(map(lambda x: int((x[0]-x[1])/2), zip(size,img.size))))
        return layer

class UNetLeaky(nn.Module):
    def __init__(self):
        super(UNetLeaky, self).__init__()

        self.c0=nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.c1=nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.c2=nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.c3=nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.c4=nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.c5=nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.c6=nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.c7=nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.c8=nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.dc8=nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dc7=nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.dc6=nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.dc5=nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.dc4=nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.dc3=nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.dc2=nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dc1=nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.dc0=nn.Conv2d(64, 3, 3, 1, 1)

        self.bnc0=nn.BatchNorm2d(32)
        self.bnc1=nn.BatchNorm2d(64)
        self.bnc2=nn.BatchNorm2d(64)
        self.bnc3=nn.BatchNorm2d(128)
        self.bnc4=nn.BatchNorm2d(128)
        self.bnc5=nn.BatchNorm2d(256)
        self.bnc6=nn.BatchNorm2d(256)
        self.bnc7=nn.BatchNorm2d(512)
        self.bnc8=nn.BatchNorm2d(512)

        self.bnd8=nn.BatchNorm2d(512)
        self.bnd7=nn.BatchNorm2d(256)
        self.bnd6=nn.BatchNorm2d(256)
        self.bnd5=nn.BatchNorm2d(128)
        self.bnd4=nn.BatchNorm2d(128)
        self.bnd3=nn.BatchNorm2d(64)
        self.bnd2=nn.BatchNorm2d(64)
        self.bnd1=nn.BatchNorm2d(32)

    def forward(self, x):
        e0 = F.leaky_relu(self.bnc0(self.c0(x)))
        e1 = F.leaky_relu(self.bnc1(self.c1(e0)))
        e2 = F.leaky_relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.leaky_relu(self.bnc3(self.c3(e2)))
        e4 = F.leaky_relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.leaky_relu(self.bnc5(self.c5(e4)))
        e6 = F.leaky_relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.leaky_relu(self.bnc7(self.c7(e6)))
        e8 = F.leaky_relu(self.bnc8(self.c8(e7)))

        d8 = F.leaky_relu(self.bnd8(self.dc8(torch.cat([e7, e8],dim=1))))
        del e7, e8
        d7 = F.leaky_relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.leaky_relu(self.bnd6(self.dc6(torch.cat([e6, d7],dim=1))))
        del d7, e6
        d5 = F.leaky_relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.leaky_relu(self.bnd4(self.dc4(torch.cat([e4, d5],dim=1))))
        del d5, e4
        d3 = F.leaky_relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.leaky_relu(self.bnd2(self.dc2(torch.cat([e2, d3],dim=1))))
        del d3, e2
        d1 = F.leaky_relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(torch.cat([e0, d1],dim=1))
        output = F.sigmoid(d0)
        del d0

        return output

def pred_single_image(img_path, model_class, checkpoint_path, img_out="assets/output.jpg", debug=False):
    # image pre-processing
    rgba_im = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGB",rgba_im.size,(255,255,255))
    bg.paste(rgba_im,(0,0),rgba_im)
    if np.max(rgba_im.size) > 200:
        bg = transforms.Scale(200)(bg)
    del rgba_im
    rgb_im = CentraliseImage(300)(bg)
    grayscale_im = rgb_im.convert("L")
    tfms = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor()
    ])
    preproc_original_image = Variable(tfms(grayscale_im).unsqueeze(0), volatile=True)
    print("==> Initializing model...")
    generator = model_class()
    checkpoint = torch.load(checkpoint_path)
    if debug:
        print(f"Using {model_class} from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
    generator.load_state_dict(checkpoint["model"])
    generator.eval()
    del checkpoint
    print("==> Coloring icon...")
    s = time.time()
    colored_image = generator(preproc_original_image)
    print(f"==> Inference took {(time.time() - s):.2f}s")
    print("==> Creating JPG...")
    combined = torch.cat([tfms(rgb_im).unsqueeze(0),colored_image.data],dim=0)
    utils.save_image(combined, img_out)
    print(f"==> Done! Check it out at {img_out}.")
################################################

parser = argparse.ArgumentParser()
parser.add_argument("icon_outline", help="path to the icon you want to convert")
args = parser.parse_args()

pred_single_image(args.icon_outline, UNetLeaky, checkpoint_path="models/outline2yellow_generator_gan.pth")
