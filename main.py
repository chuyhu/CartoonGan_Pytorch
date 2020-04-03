import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s][%(threadName)-10s] %(message)s')


import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import os
from network import Transformer
import server as apiserver
import asyncio

def get_app_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', dest='input_dir', default='test_img')
    parser.add_argument('--load-size', dest='load_size', default=450)
    parser.add_argument('--model-path', dest='model_path', default='./pretrained_model')
    parser.add_argument('--style', default='Hayao')
    parser.add_argument('--server', default=False, dest='server', action='store_true')
    parser.add_argument('--verbose', default=False, dest='verbose', action='store_true')
    parser.add_argument('--server-host', dest='server_host', default="")
    parser.add_argument('--server-port', dest='server_port', default=56841)
    parser.add_argument('--server-unix', dest='server_unix', default="")
    parser.add_argument('--output-dir', dest='output_dir', default='test_output4')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def load_model():
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
    model.eval()

    # if opt.gpu > -1:
    # 	print('GPU mode')
    # 	model.cuda()
    # else:
    # print('CPU mode')
    model.float()
    return model


def process_image(input_image):
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h *1.0 / w
    if ratio > 1:
        h = opt.load_size
        w = int(h*1.0/ratio)
    else:
        w = opt.load_size
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image

    input_image = Variable(input_image, volatile=True).float()
    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    return output_image


async def main(opts):
    if opts.verbose is not True:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(threadName)-10s] %(message)s')

    if opts.server:
        logging.info("Server mode")

        if opts.server_unix is not None and opts.server_unix != "":
            server = apiserver.APIServer(address=opts.server_unix, useUnixSocket=True)
        else:
            server = apiserver.APIServer(address=(opts.server_host, opts.server_port), useUnixSocket=False)

        await server.start()
        pass


if __name__ == '__main__':
    logging.info("PaintApp python backend loading")
    opt = get_app_arguments()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(opt))
