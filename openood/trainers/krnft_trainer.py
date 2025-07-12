import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing
from openood.networks.clip import clip
from openood.networks.clip_fixed_ood_prompt import imagenet_classes
from openood.networks.clip.clip import load, tokenize
from openood.networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import openood.utils.config as config
from torch.cuda.amp import autocast
import pdb
import time

_tokenizer = _Tokenizer()

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    
    return model    

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count   

class KRNFTTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.net = net
        self.train_loader = train_loader
        self.config = config
        # pdb.set_trace()
        self.optimizer = torch.optim.AdamW([{'params': self.net.model.taskres_prompt_learner.parameters()}
                                   ], lr=config.optimizer.lr)
        backbone = self.config.backbone.name
        self.clip_model = load_clip_to_cpu(backbone)
        self.clip_model = self.clip_model.to('cuda')
        #self.clip_model, _, _ = load(backbone, device='cuda', download_root=config.DOWNLOAD_ROOT)

        self.net.to(torch.float32)
        self.clip_model.to(torch.float32)                           

    def setup(self):
        pass 

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_meter = AverageMeter()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for train_step in tqdm(range(1,
                                        int(len(train_dataiter)/10) + 1),
                                desc='Epoch {:03d}: '.format(epoch_idx),
                                position=0,
                                leave=True,
                                disable=not comm.is_main_process()):
                batch = next(train_dataiter)
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                image_features_in, image_features_out, targets_in, targets_out, batch_image_features = \
                    self.get_in_out(self.clip_model, self.net, imagenet_classes, data, target)

                # get prompts
                logit_scale = self.net.logit_scale.exp()
                #prompt_features = self.net.get_text_features()
                
                prompt_features = self.net.model.taskres_prompt_learner(batch_image_features)
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
                prompt_features_teacher = self.net.text_features
                prompt_features_teacher = prompt_features_teacher / prompt_features_teacher.norm(dim=-1, keepdim=True)
                # image_features_in = self.net.image_taskres_prompt_learner(image_features_in)
                # image_features_out = self.net.image_taskres_prompt_learner(image_features_out)
    
                #get loss
                loss, loss_str = self.get_loss(prompt_features, prompt_features_teacher, image_features_in, image_features_out,
                                        targets_in, targets_out, logit_scale)                     

                #logits_classifier = self.net(data)
                #loss = F.cross_entropy(logits_classifier, target)
                # backward

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                loss_meter.update(loss.detach().cpu().item())
                tqdm.write(f'Train epoch:{epoch_idx}/{self.config.optimizer.num_epochs}\t'
                        f'Loss_avg:{loss_meter.avg:.6f}\t' + loss_str)

            # exponential moving average, show smooth values
            # with torch.no_grad():
            #     loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()
        if epoch_idx%1==0:
            model_save_dir = self.config.output_dir
            os.makedirs(model_save_dir, exist_ok=True)
            model_checkpoint_save_path = os.path.join(model_save_dir, 'model_checkpoint.pth')
            model_checkpoint = {
                'taskres_prompt_learner_state_dict': self.net.model.taskres_prompt_learner.state_dict(),
            }
            torch.save(model_checkpoint, model_checkpoint_save_path)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

    # def select_in_out(self, image_features, sim):
    #     idx_in = torch.topk(sim, dim=0, k=self.config.trainer.trainer_args.n_selection)[1].squeeze()
    #     image_features_crop_in_temp = torch.index_select(image_features, index=idx_in, dim=0)
    #     idx_out = torch.topk(-sim, dim=0, k=self.config.trainer.trainer_args.n_selection)[1].squeeze()
    #     image_features_crop_out_temp = torch.index_select(image_features, index=idx_out, dim=0)
    #     image_features_in_temp = image_features_crop_in_temp
    #     image_features_out_temp = image_features_crop_out_temp

    #     return image_features_in_temp, image_features_out_temp   
    
    def select_in_out(self, image_features, sim):
        idx_in = torch.topk(sim, dim=0, k=self.config.trainer.trainer_args.n_selection)[1].squeeze()
        image_features_crop_in_temp = torch.index_select(image_features, index=idx_in, dim=0)
        idx_out = torch.topk(-sim, dim=0, k=self.config.trainer.trainer_args.n_selection)[1].squeeze()
        image_features_crop_out_temp = torch.index_select(image_features, index=idx_out, dim=0)
        idx_in_top = torch.topk(sim, dim=0, k=1)[1].squeeze()
        image_features_crop_in_temp_top = torch.index_select(image_features, index=idx_in_top, dim=0)
        image_features_in_temp = image_features_crop_in_temp
        image_features_out_temp = image_features_crop_out_temp
        image_features_in_temp_top = image_features_crop_in_temp_top

        return image_features_in_temp, image_features_out_temp, image_features_in_temp_top 

    def get_in_out(self, clip, model, labels, images, targets):
        batch_image_features = []
        image_features_in = []
        image_features_out = []
        targets_in = []
        targets_out = []
        clip = clip.cuda()
        with torch.no_grad():
            for image_idx, (image, target) in enumerate(zip(images, targets)):
                label = labels[target.item()]
                # openai_imagenet_template = imagenet_templates.openai_imagenet_template
                openai_imagenet_template = [lambda c: f'a photo of a {c}.']
                select_prompts_in = [func(label) for func in openai_imagenet_template]
                
                   
                text_inputs = tokenize(select_prompts_in).cuda()
                select_prompts_in = clip.encode_text(text_inputs)
                select_prompts_in /= select_prompts_in.norm(dim=-1, keepdim=True)
                # without learnable residual or prompt

                image = image.cuda()
                target = target.cuda()
                image_features = model.get_image_features(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                

                sim = image_features @ select_prompts_in.t()
                sim = torch.max(sim, dim=1, keepdim=True)[0]

                #image_features_in_temp, image_features_out_temp = self.select_in_out(image_features, sim)
                image_features_in_temp, image_features_out_temp, image_features_in_temp_top = self.select_in_out(image_features, sim)
                image_features_in.append(image_features_in_temp)
                image_features_out.append(image_features_out_temp)
                batch_image_features.append(image_features_in_temp_top)

                # create in target
                targets_in_temp = torch.tile(target, dims=(image_features_in_temp.size(0),))
                targets_in.append(targets_in_temp)     

                # create out target
                # no use
                # prompt_features = model.get_text_features()
                # #prompt_features = model.taskres_prompt_learner()
                # prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
                # prompt_features_out = prompt_features[self.config.n_cls:, ...]
                # logit_out_temp = image_features_out_temp @ prompt_features_out.t()
                # targets_out_temp = torch.max(logit_out_temp, dim=1)[1] + self.config.n_cls
                # targets_out.append(targets_out_temp)
            
            image_features_in = torch.stack(image_features_in)
            image_features_out = torch.stack(image_features_out)
            targets_in = torch.stack(targets_in)
            batch_image_features = torch.cat(batch_image_features, dim=0)
            
            # targets_out = torch.cat(targets_out, dim=0).cuda()    

        return image_features_in, image_features_out, targets_in, targets_out, batch_image_features

    def get_loss(self, prompt_features, prompt_features_teacher, image_features_in, image_features_out, targets_in, targets_out, logit_scale):
        prompt_features_in = prompt_features[:self.config.n_cls, ...]
        prompt_features_out = prompt_features[self.config.n_cls:, ...]
        # loss_in
        
        #logit_in = logit_scale * image_features_in @ prompt_features.t()
        batch_logits_in = image_features_in @ prompt_features.t()
        batch_logits_in = logit_scale * batch_logits_in
        batch_targets_in = targets_in
        batch_targets_out = targets_out
        # loss_in = F.cross_entropy(logit_in, targets_in)
        loss_in = 0
        ngroup = self.config.n_group
        for i in range(batch_logits_in.shape[0]):
            logit_in = batch_logits_in[i]
            targets_in = batch_targets_in[i]
            id_features_in = logit_in[:, :self.config.n_cls]
            ood_features_in = logit_in[:, self.config.n_cls:]
            full_sims_list = self.grouping(id_features_in, ood_features_in, None, ngroup=ngroup, random_permute=True, softmax=False)
            full_sims_tensors = [torch.unsqueeze(tensor, dim=1) for tensor in full_sims_list]
            full_sims_tensors = torch.cat(full_sims_tensors, dim=1)
            _, max_indices = torch.max(logit_in, dim=1)
            # logit_in = logit_scale * image_features_in @ prompt_features_in.t()
            targets_expanded = targets_in.unsqueeze(1).expand(-1, len(full_sims_list))
            logit_group_in = full_sims_tensors.view(-1, full_sims_tensors.shape[-1])
            target_group_in = targets_expanded.reshape(-1)
            loss_in += F.cross_entropy(logit_group_in, target_group_in)
        loss_in = loss_in/batch_logits_in.shape[0]   

        #start group train
        batch_logits_out = image_features_out @ prompt_features.t()
        batch_logits_out = logit_scale * batch_logits_out
        loss_out = 0

        for i in range(batch_logits_out.shape[0]):
            id_features = batch_logits_out[i, :, :self.config.n_cls]
            ood_features = batch_logits_out[i, :, self.config.n_cls:]
            full_sims_list = self.grouping(id_features, ood_features, None, ngroup=ngroup, random_permute=True, softmax=True)
            # 首先对每个元素进行unsqueeze，增加一个维度
            full_sims_tensors = [torch.unsqueeze(tensor, dim=1) for tensor in full_sims_list]
            logit_out_softmax_probs_in = torch.cat(full_sims_tensors, dim=1)
            #flag_out = torch.cat([torch.FloatTensor([0] * self.config.n_cls + [1] * int(self.config.network.backbone.OOD_NUM/ngroup))], dim=0).cuda()
            flag_out = torch.cat([
                torch.zeros(self.config.n_cls, dtype=torch.float32, device='cuda'),
                torch.ones(int(self.config.network.backbone.OOD_NUM / ngroup), dtype=torch.float32, device='cuda')
            ])
            logit_out_softmax_probs_in = torch.sum(logit_out_softmax_probs_in * (1 - flag_out), dim=2) # get the image_out id score
            logit_out_softmax_probs_in_log = torch.log(logit_out_softmax_probs_in + 1e-16)
            loss_out += torch.mean(logit_out_softmax_probs_in_log) #minimize
        loss_out = loss_out/batch_logits_out.shape[0]
 
        # logit_out_softmax_probs = F.softmax(logit_out, dim=1)
        # flag_out = torch.cat([torch.LongTensor([0] * self.config.n_cls + [1] * self.config.network.backbone.n_ex_prompts)], dim=0).cuda()
        # pdb.set_trace()
        # logit_out_softmax_probs_in = torch.sum(logit_out_softmax_probs * (1 - flag_out), dim=1)
        # logit_out_softmax_probs_in_log = torch.log(logit_out_softmax_probs_in + 1e-16)
        # loss_out = torch.mean(logit_out_softmax_probs_in_log)
        
        #calculate knowledge guide loss
        cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-07)
        loss_kg = cos(prompt_features, prompt_features_teacher)
        loss_kg = 1.0-torch.mean(loss_kg)

        # loss
        loss = loss_in * self.config.lam_in + loss_out * self.config.lam_out + self.config.lam_kd*(loss_kg)
        loss_str = f'Loss_now:{loss.detach().cpu().item():.6f}\t' \
                f'Loss_in:{loss_in.detach().cpu().item():.6f}\t' \
                f'Loss_out:{loss_out.detach().cpu().item():.6f}\t' \
                f'Loss_kg:{loss_kg.detach().cpu().item():.6f}\t' 
     

        
        return loss, loss_str

    def grouping(self, pos, neg, num, ngroup=10, random_permute=False, softmax=False):
        group = ngroup
        drop = neg.shape[1] % ngroup
        if drop > 0:
            neg = neg[:, :-drop]   
        if random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg.shape[1], device="cuda:{}".format(0))
            neg = neg.T
            negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
        else:
            negs = neg.reshape(pos.shape[0], group, -1).contiguous()
        scores = []
        full_sim_list = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1)
            if softmax:
                full_sim = full_sim.softmax(dim=-1)
            full_sim_list.append(full_sim) 
        return full_sim_list    
