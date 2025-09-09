import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position

#增加自定义上采样部分代码 2025年9月5日 #增加自定义深度头 2025年9月8日

from .fppdecoder import DecoderCup, Outconv


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )
        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        
        #? fpp decoder #待验证 2025年9月6日 注意写的参数到底合不合适
        self.fppdecoder = DecoderCup( hidden_size=128, decoder_channels=(128,96,64,16), n_skip=3,skip_channels=[128,96,64,16] )
        #? fpp 深度头
        self.fppout10 = Outconv(16,1)
        self.fppout11 = Outconv(16,1)


    # 增加输出skip_feature
    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        featureall = self.backbone(concat) # list of [2B, C, H, W], resolution from high to low
        features = featureall[0]
        skip_feature = featureall[1] # skip_feature 是个list
        #↑ 由于CNN输出是个list [0]是正向特征 [1]是跳跃特征

        # reverse: resolution from low to high 
        features = features[::-1]
        skip_feature = skip_feature[::-1]

        #~ features 是一个只包含一个元素的列表。这个唯一的元素是一个四维张量，形状为 [2*B, 128, H/8, W/8] 2025年9月3日
        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # ->tuple, chunks =2 dim = 0 dim0 就是B 这个分块 
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1, skip_feature

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high #~增加输出skip_feature 2025年9月5日
        feature0_list, feature1_list,skip_feature = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales): #~num_scales=1 只循环一次
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0: 
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))
            #~ ↓ 将上一尺度的光流放大作为当前尺度的初始估计，使用双线性插值
            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None: #~ 特征扭曲
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features #~ feature_channels default=128
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            #~ 保存transformer 的输出来送到fpp的分支去，把两个特征并起来用，skip_feature 也是并起来的 在Batch维度并起来的
            featureFpp= torch.cat((feature0 , feature1), dim=0)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            #~ ↓上采样后加入到结果列表 为了效率使用最简单的双线性插值 1 上采样 2025年9月5日
            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time 
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)
            
            #~ 2 bilinear 上采样 2025年9月5日
            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)
            #~ 3 凸上采样 
            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

            #↑ 光流部分结束，↓ 下面是深度估计部分 # 2025年9月8日不知道写的对不对，反正很恶心
            #~ featureFPP 是两个feature在batch上的拼接 skip_feature 是3个feature的list 从头到尾是 从底到
            featureFpp = self.fppdecoder(featureFpp,skip_feature)
            feature10,feature11 = torch.chunk(featureFpp,chunks=2,dim=0)

            #加两个深度头 2025年9月8日
            depth10 = self.fppout10(feature10)
            depth11 = self.fppout11(feature11)


        results_dict.update({'flow_preds': flow_preds})

        return results_dict
