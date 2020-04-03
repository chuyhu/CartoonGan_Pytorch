import os

import torchvision.utils as vutils
from PIL import Image

import main

opt = main.get_app_arguments()

valid_ext = ['.jpg', '.png']
if not os.path.exists(opt.output_dir):
	os.mkdir(opt.output_dir)

# load pretrained model
model = main.load_model()

for files in os.listdir(opt.input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
	output_image = main.process_image(input_image)
	# save
	vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_' + opt.style + '.jpg'))

print('Done!')
