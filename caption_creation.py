import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


with open('caption.txt', 'a') as file:
    for i in range(100):
        img_path = f'images-2/{str(i).zfill(6)}.jpg'
        raw_image = Image.open(img_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        file.write(caption+'\n')
        print(f"Processed image {img_path} and wrote caption to file.")


