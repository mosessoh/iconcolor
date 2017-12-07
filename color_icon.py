import argparse
from momo_prod import *

parser = argparse.ArgumentParser()
parser.add_argument("icon_outline", help="path to the icon you want to convert")
args = parser.parse_args()

pred_single_image(args.icon_outline, UNetLeaky, checkpoint_path="models/outline2yellow_generator_gan.pth")
