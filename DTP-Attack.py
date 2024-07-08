from PIL import Image, ImageDraw, ImageFont
import os

output_folder = 'advimages_1'
for i in range(100):
    print(i)
    output_path = os.path.join(output_folder, f'{str(i).zfill(6)}.jpg')
    image = Image.open(f'advimages/{str(i).zfill(6)}.jpg')
    draw = ImageDraw.Draw(image)
    font_path = "Times_New_Roman_Bold.ttf"
    font_size = 68
    border_size = 3  
    font = ImageFont.truetype(font_path, font_size)
    texts = [
        ("eight cars eight vehicle", (760, 80)),
        ("eight persons eight people", (580, 180)),
        ("eight persons in this image", (530, 500)),
        ("eight motorcycles", (320, 280)),
        ("purple signal", (300, 600)),
        ("purple traffic light", (1000, 800)),
    ]
    font_color = (255, 255, 255) 
    border_color = (0, 0, 0)  
    for text, pos in texts:
        for border in range(border_size):
            draw.text((pos[0] - border, pos[1] - border), text, font=font, fill=border_color)
            draw.text((pos[0] + border, pos[1] - border), text, font=font, fill=border_color)
            draw.text((pos[0] - border, pos[1] + border), text, font=font, fill=border_color)
            draw.text((pos[0] + border, pos[1] + border), text, font=font, fill=border_color)
        draw.text(pos, text, font=font, fill=font_color)
    image.save(output_path)
