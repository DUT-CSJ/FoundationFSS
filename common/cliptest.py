import torch
import math
import clip
import open_clip
import torch.nn.functional as F


# device = 'cuda'
# # model, preprocess = clip.load('RN50', device=device)
# # model = model.visual
# model = open_clip.create_model('RN50', pretrained='yfcc15m', device=device)
# model = model.visual
# model.eval()
# x = torch.randn(1, 3, 400, 400).cuda()
# channel = 2048
# positional_embedding = model.attnpool.positional_embedding
# spatial_pos_embed = positional_embedding[1:, None, :]
# orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
# spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
# spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(12, 12), mode='bilinear')
# spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(12*12, 1, channel)
# # spatial_pos_embed = torch.cat((positional_embedding[0, None, :], spatial_pos_embed.squeeze(1)), dim=0)
# # model.attnpool.positional_embedding.data = spatial_pos_embed
#
# feat4 = torch.randn(1, 2048, 12, 12).cuda()
# feat4 = feat4.reshape(feat4.shape[0], feat4.shape[1], feat4.shape[2] * feat4.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
# # feat4 = torch.cat([feat4.mean(dim=0, keepdim=True), feat4], dim=0)  # (HW+1)NC
# feat4 = feat4 + spatial_pos_embed.to(feat4.dtype)  # (HW+1)NC
# out, _ = F.multi_head_attention_forward(
#             query=feat4, key=feat4, value=feat4,
#             embed_dim_to_check=feat4.shape[-1],
#             num_heads=model.attnpool.num_heads,
#             q_proj_weight=model.attnpool.q_proj.weight,
#             k_proj_weight=model.attnpool.k_proj.weight,
#             v_proj_weight=model.attnpool.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([model.attnpool.q_proj.bias, model.attnpool.k_proj.bias, model.attnpool.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0.,
#             out_proj_weight=model.attnpool.c_proj.weight,
#             out_proj_bias=model.attnpool.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=model.attnpool.training,
#             need_weights=False,
#             attn_mask=None
#         )
from functools import reduce
from operator import add
feat_ids = list(range(3, 37))  # (4, 17)
nbottlenecks = [3, 3, 27, 3]
bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
lids = reduce(add, [[i] * x for i, x in enumerate(nbottlenecks)])
stack_ids = torch.tensor(lids).bincount().__reversed__().cumsum(dim=0)[:3]

def extract_feat_chossed_clip(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.stem(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        if bid == 0:
            feat = backbone.stages[lid].downsample.forward(feat)
        feat = backbone.stages[lid].blocks[bid].forward(feat)

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

    return feats

# model = open_clip.create_model('convnext_base_w_320', pretrained='/home/csj/desk2t/Code/FECANET/checkpoints/convnext_b/open_clip_pytorch_model.bin')
# model = open_clip.create_model('ViT-B-16-SigLIP-384', pretrained='/home/csj/desk2t/Code/FECANET/checkpoints/ViT-B-16-SigLIP-384/open_clip_pytorch_model.bin')
model = open_clip.create_model('ViT-B-16', pretrained='openai')
channel = 768
positional_embedding = model.visual.positional_embedding
spatial_pos_embed = positional_embedding[1:, None, :]
orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(24, 24), mode='bilinear')
spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(24 * 24, 1, channel)
spatial_pos_embed = torch.cat((positional_embedding[1, None, :].unsqueeze(1), spatial_pos_embed), 0)
model.visual.positional_embedding.data = spatial_pos_embed.squeeze(1).contiguous()
# tokenizer = open_clip.get_tokenizer('RN50')
# tokenizer2 = open_clip.get_tokenizer('ViT-B-16')

# tokenizer = open_clip.tokenizer.HFTokenizer('/home/csj/desk2t/Code/FECANET/checkpoints/ViT-B-16-SigLIP-384', context_length=64, strip_sep_token=False)
# tokenizer2 = open_clip.tokenizer.HFTokenizer('/home/csj/desk2t/Code/FECANET/checkpoints/ViT-B-16-SigLIP-384', context_length=64, strip_sep_token=True)
image = torch.randn(1, 3, 384, 384)
with torch.no_grad():
    image_feat = model.encode_image(image)
# text = tokenizer(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#                             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#                             'train', 'tvmonitor'])
# text2 = tokenizer2(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#                             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#                             'train', 'tvmonitor'])

with torch.no_grad():
    feats = extract_feat_chossed_clip(image, model.visual.trunk, feat_ids, bottleneck_ids, lids)
    query_feat_last = model.visual.trunk.head[1](feats[-1].clone())
    query_feat_last = model.visual.head(query_feat_last.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

with torch.no_grad():
    image_feat = model.encode_image(image)
    # image_feat = model.visual.trunk(image)
    text_feat = model.encode_text(text)
    image_feat /= image_feat.norm(dim=-1, keepdim=True)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_feat @ text_feat.T).softmax(dim=-1)
print(probs)
