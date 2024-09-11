from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
import open_clip
import math
from einops import rearrange

from .base.conv4d import CenterPivotConv4d as Conv4d
from .base.conv4d import DWConv4d, PWConv4d


def extract_feat_chossed(img, backbone):
    r""" Extract intermediate features from vit"""
    feats = []
    feat = backbone.conv1(img)
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    feat = feat.permute(0, 2, 1)

    # B = feat.shape[0]
    # cls_tokens = backbone.class_embedding.expand(B, 1, -1)
    # feat = torch.cat((cls_tokens, feat), dim=1)
    feat = feat + backbone.positional_embedding.to(feat.dtype)

    feat = backbone.patch_dropout(feat)
    feat = backbone.ln_pre(feat)

    feat = feat.permute(1, 0, 2)  # NLD -> LND
    for i in range(backbone.transformer.layers-1):
        feat = backbone.transformer.resblocks[i](feat)
        # feats.append(feat.permute(1, 0, 2).clone())
    temp = backbone.transformer.resblocks[-1].ln_1(feat.clone())
    temp = F.linear(temp, backbone.transformer.resblocks[-1].attn.in_proj_weight, backbone.transformer.resblocks[-1].attn.in_proj_bias)
    N, L, C = temp.shape
    temp = temp.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
    temp = F.linear(temp, backbone.transformer.resblocks[-1].attn.out_proj.weight, backbone.transformer.resblocks[-1].attn.out_proj.bias)
    _, _, temp = temp.tensor_split(3, dim=0)
    temp += feat
    temp = temp + backbone.transformer.resblocks[-1].ls_2(backbone.transformer.resblocks[-1].mlp(backbone.transformer.resblocks[-1].ln_2(temp)))

    # feat = backbone.transformer.resblocks[-1](feat)
    # feats.append(feat.permute(1, 0, 2).clone())
    feats.append(temp.permute(1, 0, 2).clone())

    # feat = backbone.transformer(feat)
    # feat = feat.permute(1, 0, 2)  # LND -> NLD
    # feat = backbone.ln_post(feat)

    return feats


class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        def make_building_block_dwconv(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(DWConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(PWConv4d(inch, outch))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer4to3 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid)
        hypercorr_sqz4 = self.encoder_layer4to3(hypercorr_sqz4) + hypercorr_sqz4
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_sqz4) + hypercorr_sqz4

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)
        return logit_mask


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)
            
            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)
        corrs = torch.stack(corrs).transpose(0, 1).contiguous()
        return corrs


class HCCNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HCCNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'dino':
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            nbottlenecks = [12, 13, 13, 13]
            self.clip = open_clip.create_model('ViT-B-16', pretrained='dfn2b')
            self.clip.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
            channel = 768
            positional_embedding = self.clip.visual.positional_embedding
            spatial_pos_embed = positional_embedding[1:, None, :]
            orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
            spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
            spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(30, 30), mode='bilinear')
            spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(30 * 30, 1, channel)
            self.clip.visual.positional_embedding.data = spatial_pos_embed.squeeze(1).contiguous()
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.dino_img_mean = [0.485, 0.456, 0.406]
        self.dino_img_std = [0.229, 0.224, 0.225]
        self.dino_img_mean = torch.tensor(self.dino_img_mean).view(1, 3, 1, 1).cuda()
        self.dino_img_std = torch.tensor(self.dino_img_std).view(1, 3, 1, 1).cuda()
        self.clip_img_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_img_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_img_mean = torch.tensor(self.clip_img_mean).view(1, 3, 1, 1).cuda()
        self.clip_img_std = torch.tensor(self.clip_img_std).view(1, 3, 1, 1).cuda()

    def forward(self, query_img, support_img, support_mask, class_name):
        with torch.no_grad():
            support_img2 = torch.zeros_like(support_img)
            support_img2[:, 0, :, :] = support_img[:, 0, :, :] * support_mask
            support_img2[:, 1, :, :] = support_img[:, 1, :, :] * support_mask
            support_img2[:, 2, :, :] = support_img[:, 2, :, :] * support_mask
            support_img = support_img2
            query_feats = self.backbone.get_intermediate_layers(x=query_img, n=range(0, 12), reshape=True)
            support_feats = self.backbone.get_intermediate_layers(x=support_img, n=range(0, 12), reshape=True)
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
            
            query_clip_img = (query_img * self.dino_img_std) + self.dino_img_mean
            query_clip_img = F.interpolate(query_clip_img, size=(480, 480), mode='bilinear')
            query_clip_img = (query_clip_img - self.clip_img_mean) / self.clip_img_std
            query_clip_feat = extract_feat_chossed(query_clip_img, self.clip.visual)[0]
            query_clip_feat = self.clip.visual.ln_post(query_clip_feat)
            query_clip_feat = query_clip_feat @ self.clip.visual.proj
            query_clip_feat = rearrange(query_clip_feat, 'B (H W) C -> B C H W', H=30)
            text = self.tokenizer(class_name).cuda()
            text_feat = self.clip.encode_text(text)
            text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 30, 30)
            
            bsz, ch, hb, wb = text_feat.size()
            text_feat = text_feat.view(bsz, ch, -1)
            text_feat = text_feat / (text_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)
            
            bsz, ch, ha, wa = query_clip_feat.size()
            query_clip_feat = query_clip_feat.view(bsz, ch, -1)
            query_clip_feat = query_clip_feat / (query_clip_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)
            
            corrt = torch.bmm(query_clip_feat.transpose(1, 2), text_feat).view(bsz, ha, wa, hb, wb)
            corrt = corrt.clamp(min=0).unsqueeze(1)

            corr = torch.cat((corr, corrt), dim=1)

        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx],  batch['class_name'])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
