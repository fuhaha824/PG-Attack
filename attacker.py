import numpy as np 
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class NewAttacker():
    def __init__(self, model, multi_attacker):
        self.model = model
        self.multi_attacker = multi_attacker

    def attack(self, j, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        momentum = torch.zeros_like(imgs).detach().to(device)
        # mask_image path
        img_path = f'images_mask/{str(j).zfill(6)}.jpg'
        images_0 = Image.open(img_path)
        transform = transforms.ToTensor()
        mask_tensor = transform(images_0)
        mask_tensor = mask_tensor.to(device)
        steps = 5
        momentum = self.multi_attacker.pre_attack(self.model, txts, imgs,mask_tensor,txt2img, steps, momentum, scales=scales)
        steps = 60
        adv_imgs = imgs
        adv_imgs = self.multi_attacker.img_attack(self.model, txts, adv_imgs,imgs,mask_tensor,txt2img, steps, momentum,device, scales=scales)

        return adv_imgs
 

class Attacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30,number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32, imgs_eps=128/255, step_size=2/255):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.imgs_eps = imgs_eps 
        self.step_size = step_size

    def pre_attack(self, net, texts,imgs,mask_tensor,txt2img,steps,momentum, scales=None):
        device = self.ref_net.device
        b, _, _, _ = imgs.shape
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1
        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        imgs_0 = imgs.detach()* (1-mask_tensor) + torch.from_numpy(np.random.uniform(-self.imgs_eps, self.imgs_eps, imgs.shape)).float().to(device) * mask_tensor
        imgs_0 = imgs.detach().to(device)
        imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)

        perturbation = imgs.detach() * mask_tensor

        for _ in range(steps):

            imgs_0.requires_grad_()
            imgs_1 = F.interpolate(imgs_0,size=(384,384),mode='bilinear', align_corners=False)
            imgs_output = net.inference_image(images_normalize(imgs_1))
            imgs_embeds = imgs_output['image_feat'][txt2img]
            net.zero_grad()
            txts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
            txts_output = net.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

            scaled_imgs = self.get_scaled_imgs(imgs_1, scales, device)     
            imgs_output = net.inference_image(images_normalize(scaled_imgs))

            imgs_embeds = imgs_output['image_feat']
            with torch.enable_grad():
                img_loss_list = []
                img_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    img_loss_item = self.loss_func1(imgs_embeds[i*b:i*b+b], txt_supervisions, txt2img)
                    img_loss_list.append(img_loss_item.item())
                    img_loss += img_loss_item

            loss = img_loss
            loss.backward()
            grad = imgs_0.grad
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            grad = grad + momentum * 0.9
            momentum = grad
            perturbation =  10 * self.step_size * grad.sign()
            imgs_0_adv = imgs_0.detach() + perturbation
            imgs_0 = imgs_0.detach() * mask_tensor + imgs_0_adv * (1 - mask_tensor)
            imgs_0 = torch.min(torch.max(imgs_0, imgs - self.imgs_eps), imgs + self.imgs_eps)
            imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)
        return momentum
    


    def img_attack(self, model, texts, imgs,origin_imgs,mask_tensor,txt2img,steps,momentum,device, scales=None):
        model.eval()
        b, _, _, _ = imgs.shape
    
        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        imgs_0 = imgs.detach()* (1-mask_tensor) + torch.from_numpy(np.random.uniform(-self.imgs_eps, self.imgs_eps, origin_imgs.shape)).float().to(device) * mask_tensor
        imgs_0 = imgs.detach().to(device)
        
        imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)
        txts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
        txts_output = model.inference_text(txts_input)
        txt_supervisions = txts_output['text_feat'].detach()
        mix_scales = [1, 1 / 2, 1 / 4,1 / 8,  1 / 16]
        perturbation = imgs.detach() * mask_tensor

        for j in range(steps):
            imgs_0.requires_grad_()
            model.zero_grad()
            imgs_1 = F.interpolate(imgs_0,size=(384,384),mode='bilinear', align_corners=False)
            image_aug = self.get_scaled_imgs(imgs_1, scales, device)  
            image_aug = self.input_diversity(image_aug)

            image_aug = torch.cat([image_aug * scale for scale in mix_scales])
            image_aug = torch.cat((image_aug, imgs_1), dim=0)
            scaled_imgs = image_aug
            imgs_output = model.inference_image(images_normalize(scaled_imgs))
            imgs_embeds = imgs_output['image_feat']
            with torch.enable_grad():
                img_loss_list = []
                img_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(26):
                    img_loss_item = self.loss_func1(imgs_embeds[i*b:i*b+b], txt_supervisions, txt2img)
                    img_loss_list.append(img_loss_item.item())
                    img_loss += img_loss_item

            loss =  img_loss 
            loss.backward()
            grad = imgs_0.grad
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            grad = grad + momentum * 0.9
            momentum = grad
            perturbation =  self.step_size * grad.sign()
            imgs_0_adv = imgs_0.detach() + perturbation
            imgs_0 = imgs_0.detach() * mask_tensor + imgs_0_adv * (1 - mask_tensor)
            imgs_0 = torch.min(torch.max(imgs_0, origin_imgs - self.imgs_eps), origin_imgs + self.imgs_eps)
            imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)
        return imgs_0

    def input_diversity(self, img):
        size = img.size(2)
        resize = int(size / 0.875)
        rnd = torch.randint(size, resize + 1, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
        h_rem = resize - rnd
        w_rem = resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,)).item()
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (size, size), mode="nearest")
        return padded
    
    def loss_func1(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)



