
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device) # CLIP.png为本文中图一，即CLIP的流程图
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) # 将这三句话向量化

with torch.no_grad():
    # image_features = model.encode_image(image) # 将图片进行编码
    # text_features = model.encode_text(text)    # 将文本进行编码
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)