import argparse
import os
import ruamel.yaml 
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip
import utils
from attacker import NewAttacker, Attacker

advimages_path = './advimages'
def retrieval_eval(model, ref_model, tokenizer, device):
    model.float()
    model.eval()
    ref_model.eval()
    multi_attacker = Attacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3)
    attacker = NewAttacker(model, multi_attacker)
    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
    else:
        scales = None    
    toPIL = transforms.ToPILImage()
    # caption
    with open('caption.txt','r') as text_file: 
        for j in range(100):
            print(j)
            txt2img = []
            texts = []
            text = text_file.readline().strip()
            for i in range (3):
                texts.append(text)
            txt2img = [0,0,0]
            # 待攻击图像路径
            img_path = f'images-2/{str(j).zfill(6)}.jpg'

            images_0 = Image.open(img_path)
            transform = transforms.ToTensor()
            image_tensor = transform(images_0)
            images = image_tensor.to(device)  
            images = images.unsqueeze(0)         
            adv_images = attacker.attack(j, images, texts, txt2img, device=device,
                                                    max_lemgth=30, scales=scales) 
            img_PIL = toPIL(adv_images.squeeze(0))
            img_PIL.save(os.path.join(advimages_path, f'{str(j).zfill(6)}.jpg'))

    return  0


def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    else:
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer


def main(args, config):
    device = torch.device('cuda')
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args.source_model, args.source_ckpt, args.source_text_encoder, device)
    model = model.to(device)
    ref_model = ref_model.to(device)
    retrieval_eval(model, ref_model, tokenizer, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--source_model', default='TCL', type=str)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
    parser.add_argument('--source_ckpt', default='./checkpoint/tcl_retrieval_flickr.pth', type=str)
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    args = parser.parse_args()
    yaml = ruamel.yaml.YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))
    main(args, config)