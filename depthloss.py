import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ==============================================================================
# 1. L1 损失 (L1 Loss)
# ==============================================================================
# 描述：L1损失，也称为最小绝对误差（MAE），计算预测值和目标值之间差异的绝对值的平均值。
# 它对异常值不那么敏感，是监督式深度估计中常用的对齐项。
# PyTorch 提供了内置实现。

# 使用方法:
# loss_fn = nn.L1Loss()
# loss = loss_fn(prediction, target)


# ==============================================================================
# 2. BerHu 损失 (Reversed Huber Loss)
# ==============================================================================
# 描述：BerHu 损失是 L1 和 L2 损失的混合体。对于较小的误差，它使用 L2 惩罚；
# 对于超过阈值 c 的较大误差，它使用 L1 惩罚。这使得它在保持对大误差鲁棒性的同时，
# 对小误差的惩罚也更为平滑。

class BerHuLoss(nn.Module):
    """
    BerHu 损失的实现。
    """
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测的深度图 (B, C, H, W)。
            target (torch.Tensor): 真实的深度图 (B, C, H, W)。
        """
        assert pred.dim() == target.dim(), "输入和目标的维度必须相同。"
        
        # 通常在深度估计中，我们只对有效的（非零）目标像素计算损失
        valid_mask = (target > 0).detach()
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        diff = torch.abs(pred_valid - target_valid)
        
        # 根据批次中的最大误差动态确定阈值 c
        delta = self.threshold * torch.max(diff).item()

        # 计算 L1 和 L2 部分
        l1_part = diff
        l2_part = (diff**2 + delta**2) / (2 * delta)
        
        # 根据误差是否超过阈值来选择损失
        loss = torch.where(diff > delta, l1_part, l2_part)
        
        return loss.mean()


# ==============================================================================
# 3. 梯度匹配损失 (L_gm / Gradient Matching Loss)
# ==============================================================================
# 描述：此损失函数作为正则化项，强制要求预测深度图的局部梯度与真实深度图的梯度相匹配。
# 这有助于模型学习到更锐利的物体边界和平滑的表面细节。

class GradientMatchingLoss(nn.Module):
    """
    梯度匹配损失 (L_gm) 的实现。
    """
    def __init__(self):
        super(GradientMatchingLoss, self).__init__()
        # 使用 Sobel 算子计算图像梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 将 Sobel 算子注册为 buffer，这样它们会被移动到正确的设备（CPU/GPU）
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测的深度图 (B, 1, H, W)。
            target (torch.Tensor): 真实的深度图 (B, 1, H, W)。
        """
        # 计算预测图的梯度
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding='same')
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding='same')

        # 计算目标图的梯度
        target_grad_x = F.conv2d(target, self.sobel_x, padding='same')
        target_grad_y = F.conv2d(target, self.sobel_y, padding='same')

        # 计算梯度差异的 L1 损失
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        # 只在有效像素上计算损失
        valid_mask = (target > 0).detach()
        loss = (grad_diff_x[valid_mask].mean() + grad_diff_y[valid_mask].mean())
        
        return loss


# ==============================================================================
# 4. 边缘感知平滑度损失 (Edge-Aware Smoothness Loss)
# ==============================================================================
# 描述：这是自监督学习中一个关键的正则化项。它鼓励预测的深度图是分段平滑的，
# 即只在图像中存在强边缘（高梯度）的区域允许深度的剧烈变化。

class EdgeAwareSmoothnessLoss(nn.Module):
    """
    边缘感知平滑度损失的实现。
    """
    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()
        # 使用简单的差分计算梯度
    
    def forward(self, disp, img):
        """
        Args:
            disp (torch.Tensor): 预测的视差图 (B, 1, H, W)。在自监督中通常预测视差。
            img (torch.Tensor): 对应的输入 RGB 图像 (B, 3, H, W)。
        """
        # 计算视差图的梯度
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # 计算图像的梯度
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        # 计算边缘感知权重
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # 返回加权梯度的平均值
        return grad_disp_x.mean() + grad_disp_y.mean()


# ==============================================================================
# 5. 结构相似性损失 (SSIM Loss)
# ==============================================================================
# 描述：SSIM 衡量两张图像在结构、亮度和对比度上的相似性。作为损失函数时，
# 通常目标是最大化 SSIM，等价于最小化 (1 - SSIM)。这在自监督学习中是光度
# 重建损失的核心组成部分，在监督学习中也可用作正则化项。

class SSIMLoss(nn.Module):
    """
    结构相似性 (SSIM) 损失的实现。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        # 计算 SSIM 值，范围在 [-1, 1]
        ssim_val = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
        
        # 将 SSIM 转换为损失，范围在 
        # (1 - ssim) / 2 是自监督学习中常用的形式
        return (1.0 - ssim_val) / 2.0

if __name__ == '__main__':
    # 示例：如何使用这些损失函数
    
    # 创建虚拟数据
    # 假设批次大小为 4，1 个通道（灰度图），高度 64，宽度 64
    pred_depth = torch.rand(4, 1, 64, 64)
    true_depth = torch.rand(4, 1, 64, 64)
    
    # 在真实深度图中设置一些无效像素（值为0）
    true_depth[:, :, 10:20, 10:20] = 0
    
    # 对应的 RGB 图像
    rgb_image = torch.rand(4, 3, 64, 64)

    # 1. L1 损失
    l1_loss_fn = nn.L1Loss()
    l1_loss = l1_loss_fn(pred_depth[true_depth > 0], true_depth[true_depth > 0])
    print(f"L1 Loss: {l1_loss.item()}")

    # 2. BerHu 损失
    berhu_loss_fn = BerHuLoss(threshold=0.2)
    berhu_loss = berhu_loss_fn(pred_depth, true_depth)
    print(f"BerHu Loss: {berhu_loss.item()}")

    # 3. 梯度匹配损失
    gm_loss_fn = GradientMatchingLoss()
    gm_loss = gm_loss_fn(pred_depth, true_depth)
    print(f"Gradient Matching Loss: {gm_loss.item()}")

    # 4. 边缘感知平滑度损失
    # 注意：此损失通常用于自监督，输入是视差图和RGB图像
    pred_disp = 1.0 / (pred_depth + 1e-6) # 简单转换为视差
    smoothness_loss_fn = EdgeAwareSmoothnessLoss()
    smoothness_loss = smoothness_loss_fn(pred_disp, rgb_image)
    print(f"Edge-Aware Smoothness Loss: {smoothness_loss.item()}")

    # 5. SSIM 损失
    ssim_loss_fn = SSIMLoss()
    ssim_loss = ssim_loss_fn(pred_depth, true_depth)
    print(f"SSIM Loss: {ssim_loss.item()}")

