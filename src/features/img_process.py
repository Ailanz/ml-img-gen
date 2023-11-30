import glob
import numpy as np
import torch
from PIL import Image
import open_clip
import requests
from io import BytesIO

img_path = '../../data/pokemon_aug/*.jpeg'


def load_imgs(path=img_path):
    filelist = glob.glob(path)
    # filelist = glob.glob('../../data/pokemon_jpg/*.jpg')

    x = [Image.open(fname) for fname in filelist]
    print('Image Shape: ', len(x))
    return x


def load_imgs_as_np(path = img_path):
    filelist = glob.glob(path)
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    print('Image Shape: ', x.shape)
    return x


def img_embedding(imgs_arr=[], text=[]):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image_list = [preprocess(img).unsqueeze(0) for img in imgs_arr]
    concat_image = torch.cat(image_list, dim=0)
    text = tokenizer(text)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(concat_image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        print('Clip Shape: ', image_features.shape)
        return image_features.numpy(), text_features.numpy()
        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


# print(torch.cuda.is_available())
# imgs = load_imgs(as_np=False)
# img_embedding(imgs_arr=imgs, text=['a dog', 'a cat'])
