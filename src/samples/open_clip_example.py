import torch
from PIL import Image
import open_clip
import requests
from io import BytesIO

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32B_b79k')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

url = "https://d2zp5xs5cp8zlg.cloudfront.net/image-78806-800.jpg"
labels = ["a diagram", "a kitten",  "a dog", "a cat"]
# load img from url
response = requests.get(url)
img = Image.open(BytesIO(response.content))
image = preprocess(img).unsqueeze(0)
text = tokenizer(labels)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print image feature shape
print(image_features.shape)
# print text feature shape
print(text_features.shape)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
print("Label: ", labels[text_probs.argmax(dim=-1)])  # prints: [0]
