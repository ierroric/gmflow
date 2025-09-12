import torch


def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics



def berhu_loss(depth_pred, depth_gt, threshold=0.2):
    """
    BerHu 损失函数的实现。
    Args:
        pred (torch.Tensor): 预测的深度图 (B, C, H, W)
        target (torch.Tensor): 真实的深度图 (B, C, H, W)
        threshold (float): 决定何时从L1切换到L2的阈值比例
    Returns:
        torch.Tensor: 计算得到的BerHu损失
    """
    # 创建有效掩码（只处理非零目标值）
    valid_mask = (depth_gt > 0)
    
    # 应用掩码 2025年9月11日在这里这样使用掩码会把tensor隐性拉直
    depth_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]
    
    # 计算绝对误差
    diff = torch.abs(depth_valid- gt_valid)
    
    # 动态确定阈值c（保持为张量以保留梯度）
    delta = threshold * torch.max(diff)
    
    # 避免除零错误（添加一个小 epsilon）
    delta = torch.clamp(delta, min=1e-6)
    
    # 根据BerHu公式计算损失
    # 当误差小于等于delta时，使用L1损失
    # 当误差大于delta时，使用L2形式的损失
    loss = torch.where(
        diff <= delta,
        diff,  # L1部分
        (diff**2 + delta**2) / (2 * delta)  # L2部分
    )
    
    return loss.mean()
