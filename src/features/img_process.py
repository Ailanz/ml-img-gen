import glob
import numpy as np
import torch
from PIL import Image
import open_clip
import requests
from io import BytesIO

def load_imgs(as_np = True):
    filelist = glob.glob('../../data/pokemon_jpg/*.jpg')
    x = []
    if as_np:
        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        print('Image Shape: ', x.shape)
    else:
        x = [Image.open(fname) for fname in filelist]
        print('Image Shape: ', len(x))
    return x


def img_embedding(imgs = [], text = []):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32B_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    image = preprocess(imgs).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        print('Clip Shape: ', image_features.shape)
        return image_features, text_features
        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


imgs = load_imgs(as_np=False)
img_embedding(imgs=imgs, text=['a dog', 'a cat'])
