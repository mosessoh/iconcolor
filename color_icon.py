import argparse
from momo_imports import *
import momo_utils as mu

parser = argparse.ArgumentParser()
parser.add_argument("icon_outline", help="path to the icon you want to convert")
args = parser.parse_args()

mu.pred_single_image(args.icon_outline, mu.UNetLeaky, checkpoint_path="models/outline2yellow_generator_gan.pth")
