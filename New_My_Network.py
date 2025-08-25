import torch
import statistics
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation, \
    ToTensor, ToPILImage
from torchvision.models import vgg16
from einops import rearrange, repeat
import numpy as np
from PIL import Image
import os
import time
import cv2
import matplotlib.pyplot as plt
from torchprofile import profile_macs

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 自定义数据集加载器（增强颜色相关变换）
class LowLightDataset(Dataset):
    def __init__(self, low_dir="F:/研究生论文/10_image_enhanced/LOLdataset/train/low/",
                 high_dir="F:/研究生论文/10_image_enhanced/LOLdataset/train/high/", train=True):
        self.low_paths = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])
        self.high_paths = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir)])
        self.train = train

        # 训练数据增强 - 强化颜色变换
        self.train_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1),  # 增强颜色扰动
            RandomRotation(15),
            ToTensor()
        ])

        # 验证/测试转换
        self.eval_transform = Compose([
            ToTensor()
        ])

    def __len__(self):
        return min(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert('RGB')
        high_img = Image.open(self.high_paths[idx]).convert('RGB')

        # 应用数据增强（如果是训练模式）
        if self.train:
            # 对低光照和高光照图像应用相同的随机变换
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            low_img = self.train_transform(low_img)
            torch.manual_seed(seed)
            high_img = self.train_transform(high_img)
        else:
            low_img = self.eval_transform(low_img)
            high_img = self.eval_transform(high_img)

        return low_img, high_img


# 2. 多头注意力模块（优化）
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(4, dim)  # 添加归一化
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out) + x  # 添加残差连接


# 3. 颜色注意力模块（新增）
class ColorAttention(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.color_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 3, 1),  # 输出RGB三个通道的注意力图
            nn.Sigmoid()
        )
        self.conv3_64 = nn.Conv2d(3, in_channels, 3, padding=1)

    def forward(self, x):
        # 生成颜色注意力图
        color_weights = self.color_attn(x)
        color_weights = self.conv3_64(color_weights)

        # print("color_weights", color_weights.size())
        # print("x", x.size())

        # 应用颜色注意力
        return x * color_weights, color_weights


# 4. 带多头注意力的Transformer骨干（添加颜色注意力）
class TransformerBackbone(nn.Module):
    def __init__(self, dim=64, depth=4, heads=4):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=8, stride=8),
            nn.GroupNorm(4, dim),  # 添加归一化
            nn.GELU()
        )

        # 创建Transformer块序列（添加颜色注意力）
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(4, dim),
                    nn.GELU()
                ),
                MultiHeadAttention(dim, heads),
                ColorAttention(dim),  # 新增颜色注意力
                nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(4, dim)
                )
            ])
            self.blocks.append(block)

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            residual = x

            # 第一卷积层
            x = block[0](x)

            # 注意力层
            x = block[1](x)

            # 颜色注意力
            x, color_attn = block[2](x)

            # 第二卷积层
            x = block[3](x)

            # 残差连接
            x = x + residual

        return x, color_attn


# 5. 带注意力机制的光照处理与融合（添加颜色校正分支）
class AttLightFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 光照图估计
        self.illum_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # 自适应融合的注意力机制
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(64 + 3, 16, 3, padding=1),  # 简化输入通道
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]), requires_grad=True)

        # 颜色校正分支
        self.color_correction = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

        self.conv6_67 = nn.Conv2d(in_channels=6, out_channels=67, kernel_size=3, stride=1, padding=1)
        self.conv3_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, stage_feat, low_img):
        # 路径1: 增强结果与Gamma融合
        gamma05 = low_img ** 0.5
        gamma03 = low_img ** 0.3

        # 使用注意力权重融合

        stage_feat = F.interpolate(stage_feat, scale_factor=2, mode='bilinear', align_corners=False)
        attn_input = torch.cat([stage_feat, gamma05], dim=1)
        # print("gamma05", gamma05.size())
        # print("stage_feat", stage_feat.size())
        attn_input = self.conv6_67(attn_input)
        # print("attn_input", attn_input.size())
        attn_weights = self.fusion_attn(attn_input)

        # 融合增强特征和Gamma校正图像
        p1 = attn_weights * stage_feat + (1 - attn_weights) * (0.7 * gamma05 + 0.3 * gamma03)

        # 路径2: 光照图反馈
        stage_feat = self.conv3_64(stage_feat)
        illum_map = self.illum_conv(stage_feat)
        p2 = low_img * (1.0 + 0.5 * illum_map)  # 调整光照增强强度

        # 使用可学习权重进行最终融合
        weights = torch.softmax(self.fusion_weight, dim=0)
        fused_img = weights[0] * p1 + weights[1] * p2

        # 应用颜色校正
        color_corrected = self.color_correction(fused_img)
        final_img = fused_img * color_corrected

        return final_img, illum_map, color_corrected


# 6. 多阶段增强网络（带Transformer和残差连接）
class MultiStageTransformer(nn.Module):
    def __init__(self, num_stages=3):
        super().__init__()
        self.transformer = TransformerBackbone()

        # 多阶段处理模块（添加残差连接）
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(4, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(4, 64)
            ) for _ in range(num_stages)
        ])

        # 上采样到原始分辨率
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, low_img):
        x, color_attn = self.transformer(low_img)

        # 多阶段处理
        for stage in self.stages:
            residual = x
            x = stage(x)
            x = x + residual  # 残差连接

        # 上采样到原始分辨率
        x = self.upsample(x)
        return x, color_attn


# 7. 完整模型架构（添加颜色增强）
class LowLightEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhancer = MultiStageTransformer()
        self.fusion = AttLightFusion()

    def forward(self, low_img):
        # 增强处理
        enhanced_feat, color_attn = self.enhancer(low_img)

        # 双路径融合（返回颜色校正图）
        final_img, illum_map, color_corrected = self.fusion(enhanced_feat, low_img)
        return final_img, illum_map, color_corrected, color_attn


# 8. 混合损失函数（添加颜色一致性损失）
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        # 加载VGG16用于感知损失
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.perceptual = nn.Sequential(*vgg).to(device)

        # 颜色直方图损失
        self.hist_loss = nn.MSELoss()

        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def compute_color_histogram(self, img, bins=64):
        """计算RGB图像的直方图"""
        histograms = []
        for c in range(3):
            channel = img[:, c, :, :]
            hist = torch.histc(channel, bins=bins, min=0.0, max=1.0)
            hist = hist / hist.sum()  # 归一化
            histograms.append(hist)
        return torch.cat(histograms, dim=0)

    def color_consistency_loss(self, pred, target):
        """计算颜色一致性损失"""
        pred_hist = self.compute_color_histogram(pred)
        target_hist = self.compute_color_histogram(target)
        return self.hist_loss(pred_hist, target_hist)

    def forward(self, pred, target, illum_map):
        # 重建损失
        loss_l1 = self.l1(pred, target)
        loss_mse = self.mse(pred, target)

        # 光照平滑损失
        grad_x = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1])
        grad_y = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :])
        loss_smooth = (grad_x.mean() + grad_y.mean()) * 0.5

        # 感知损失（使用VGG16特征）
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        percep_loss = F.l1_loss(self.perceptual(pred_norm), self.perceptual(target_norm))

        # 颜色一致性损失
        color_loss = self.color_consistency_loss(pred, target)

        # 组合损失
        return (0.4 * loss_l1 + 0.2 * loss_mse + 0.2 * loss_smooth +
                0.1 * percep_loss + 0.1 * color_loss)


# 9. 训练函数（添加可视化）
def train_model(epochs=100, batch_size=1, lr=1e-4):
    # 初始化
    model = LowLightEnhancer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = HybridLoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 数据集
    train_dataset = LowLightDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 验证集
    val_dataset = LowLightDataset(train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        # 训练阶段
        for i, (low_imgs, high_imgs) in enumerate(train_loader):
            low_imgs, high_imgs = low_imgs.to(device), high_imgs.to(device)

            # 混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                enhanced, illum_map, color_corrected, color_attn = model(low_imgs)

                # 计算损失
                loss = criterion(enhanced, high_imgs, illum_map)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 打印批次信息
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

                # 可视化示例（每50个批次）
                if (i + 1) % 50 == 0:
                    visualize_results(low_imgs[0], enhanced[0], high_imgs[0],
                                      color_attn[0], color_corrected[0],
                                      epoch, i, "train")

        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, (low_imgs, high_imgs) in enumerate(val_loader):
                low_imgs, high_imgs = low_imgs.to(device), high_imgs.to(device)
                enhanced, illum_map, color_corrected, color_attn = model(low_imgs)
                loss = criterion(enhanced, high_imgs, illum_map)
                val_loss += loss.item()

                # 可视化验证结果（第一个批次）
                if j == 0:
                    visualize_results(low_imgs[0], enhanced[0], high_imgs[0],
                                      color_attn[0], color_corrected[0],
                                      epoch, j, "val")

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)  # 根据验证损失调整学习率

        # 打印epoch信息
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{epochs}] Complete, Time: {epoch_time:.1f}s")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_enhancer.pth")
            print(f"Saved best model with val loss: {avg_val_loss:.4f}")

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"enhancer_epoch_{epoch + 1}.pth")

    print("Training Complete!")
    torch.save(model.state_dict(), "final_enhancer.pth")
    return model


# 修改后的 visualize_results 函数
def visualize_results(low_img, enhanced, target, color_attn, color_corrected, epoch, batch, phase):
    # 转换为numpy并确保数据类型为float32
    low_img = low_img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    enhanced = enhanced.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    target = target.cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    # 处理颜色注意力图
    color_attn = color_attn.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # [H, W, C]
    color_attn = (color_attn - color_attn.min()) / (color_attn.max() - color_attn.min())

    # 取通道均值，得到单通道的注意力图
    color_attn = color_attn.mean(axis=2)  # 现在形状是 [H, W]
    color_attn = np.clip(color_attn, 0, 1)  # 归一化到 [0, 1]

    # 处理颜色校正图
    color_corrected = color_corrected.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # [H, W, C]
    color_corrected = (color_corrected - color_corrected.min()) / (color_corrected.max() - color_corrected.min())

    # 创建图像
    plt.figure(figsize=(20, 10))

    # 原始低光照图像
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(low_img, 0, 1))
    plt.title("Low Light Input")
    plt.axis('off')

    # 增强结果
    plt.subplot(2, 3, 2)
    plt.imshow(np.clip(enhanced, 0, 1))
    plt.title("Enhanced Result")
    plt.axis('off')

    # 目标图像
    plt.subplot(2, 3, 3)
    plt.imshow(np.clip(target, 0, 1))
    plt.title("Target")
    plt.axis('off')

    # 颜色注意力图
    plt.subplot(2, 3, 4)
    plt.imshow(color_attn, cmap='viridis')
    plt.title("Color Attention Map")
    plt.axis('off')

    # 颜色校正图
    plt.subplot(2, 3, 5)
    plt.imshow(np.clip(color_corrected, 0, 1))
    plt.title("Color Correction")
    plt.axis('off')

    # 颜色直方图比较
    plt.subplot(2, 3, 6)
    for i, color in enumerate(['r', 'g', 'b']):
        plt.hist(enhanced[:, :, i].flatten(), bins=50, alpha=0.5, color=color, label=f"Enhanced {color.upper()}")
        plt.hist(target[:, :, i].flatten(), bins=50, alpha=0.3, color=color, label=f"Target {color.upper()}",
                 histtype='step', linewidth=2)
    plt.title("Color Histograms")
    plt.legend()

    # 保存图像
    os.makedirs(f"visualizations/{phase}", exist_ok=True)
    plt.savefig(f"visualizations/{phase}/epoch_{epoch + 1}_batch_{batch + 1}.png", bbox_inches='tight')
    plt.close()


# 10. 推理函数（支持批处理和HDR输出）
def enhance_image(model, low_img_path, output_path=None, hdr_output=False):
    # 加载图像
    if isinstance(low_img_path, list):
        # 批量处理
        low_imgs = [Image.open(path).convert('RGB') for path in low_img_path]
        low_tensors = torch.stack([ToTensor()(img) for img in low_imgs]).to(device)
    else:
        # 单图像处理
        low_img = Image.open(low_img_path).convert('RGB')
        low_tensors = ToTensor()(low_img).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        enhanced, _, color_corrected, color_attn = model(low_tensors)

    # 处理结果
    results = []
    for i in range(enhanced.shape[0]):
        img = enhanced[i].cpu().clamp(0, 1)

        # 转换为PIL图像
        enhanced_img = ToPILImage()(img) if hasattr(ToPILImage, '__call__') else \
            Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        # 保存结果
        if output_path:
            if isinstance(output_path, list):
                save_path = output_path[i]
            else:
                save_path = output_path if i == 0 else f"{os.path.splitext(output_path)[0]}_{i}{os.path.splitext(output_path)[1]}"

            enhanced_img.save(save_path)

            # 保存HDR版本
            if hdr_output:
                hdr_path = f"{os.path.splitext(save_path)[0]}_hdr.hdr"
                hdr_img = img.permute(1, 2, 0).numpy() * 6.0  # 扩展动态范围
                cv2.imwrite(hdr_path, cv2.cvtColor(hdr_img, cv2.COLOR_RGB2BGR))

        # 保存颜色信息
        color_attn_img = color_attn[i].cpu().clamp(0, 1)
        color_attn_img = ToPILImage()(color_attn_img)
        color_attn_img.save(f"{os.path.splitext(save_path)[0]}_color_attn.jpg")

        color_corrected_img = color_corrected[i].cpu().clamp(0, 1)
        color_corrected_img = ToPILImage()(color_corrected_img)
        color_corrected_img.save(f"{os.path.splitext(save_path)[0]}_color_corrected.jpg")

        results.append(enhanced_img)

    return results if len(results) > 1 else results[0]


# 11. 模型量化函数
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


# 12. ONNX导出函数
def export_to_onnx(model, output_path="lowlight_enhancer.onnx"):
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output', 'illum_map', 'color_corrected', 'color_attn'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'illum_map': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'color_corrected': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'color_attn': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    print(f"Model exported to {output_path}")


def analyze_model_complexity(model, input_size=(1, 3, 256, 256)):
    """分析模型的参数量和计算复杂度"""
    # 创建虚拟输入
    dummy_input = torch.randn(*input_size).to(device)

    # 方法1: 使用PyTorch内置方法计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总量: {total_params:,}")

    # 方法2: 使用thop计算FLOPs
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"THOP计算 - FLOPs: {flops:,} 参数: {params:,}")

    # 方法3: 使用torchprofile计算MACs
    macs = profile_macs(model, dummy_input)
    print(f"MACs: {macs:,} (约等于 {macs / 1e6:.2f} MMACs)")

    return total_params, flops, macs


# 15. 推理速度测试函数
def measure_inference_speed(model, input_size=(1, 3, 256, 256), repetitions=100, warmup=10):
    """测量模型的推理速度"""
    # 创建虚拟输入
    dummy_input = torch.randn(*input_size).to(device)

    # 预热阶段
    print(f"预热运行 ({warmup}次)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    # 正式测量
    print(f"测量推理速度 ({repetitions}次)...")
    timings = []
    for i in range(repetitions):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)  # 转换为毫秒

        # 每10次打印一次进度
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{repetitions} 次推理")

    # 计算统计信息
    avg_time = statistics.mean(timings)
    std_dev = statistics.stdev(timings)
    min_time = min(timings)
    max_time = max(timings)

    print("\n推理速度分析结果:")
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"标准差: {std_dev:.2f} ms")
    print(f"最小时间: {min_time:.2f} ms")
    print(f"最大时间: {max_time:.2f} ms")
    print(f"帧率(FPS): {1000 / avg_time:.2f}")

    # 返回结果
    return {
        'avg_time': avg_time,
        'std_dev': std_dev,
        'min_time': min_time,
        'max_time': max_time,
        'fps': 1000 / avg_time
    }


# 16. 模型分析报告函数
def generate_model_report(model):
    """生成完整的模型分析报告"""
    print("=" * 60)
    print("模型复杂度分析")
    print("=" * 60)
    params, flops, macs = analyze_model_complexity(model)

    print("\n" + "=" * 60)
    print("推理速度分析")
    print("=" * 60)
    speed_data = measure_inference_speed(model)

    # 生成报告
    report = {
        'parameters': params,
        'flops': flops,
        'macs': macs,
        'inference_speed': speed_data
    }

    print("\n" + "=" * 60)
    print("模型分析摘要")
    print("=" * 60)
    print(f"参数量: {params:,}")
    print(f"FLOPs: {flops:,} ({flops / 1e9:.2f} GFLOPs)")
    print(f"MACs: {macs:,} ({macs / 1e6:.2f} MMACs)")
    print(f"平均推理时间: {speed_data['avg_time']:.2f} ms")
    print(f"帧率(FPS): {speed_data['fps']:.2f}")

    return report


# 修改主函数，在训练后添加模型分析
if __name__ == "__main__":
    # 开始训练
    model = train_model(epochs=2000, batch_size=8)

    # 导出模型
    export_to_onnx(model)

    # 量化模型
    quantized_model = quantize_model(model)
    torch.save(quantized_model.state_dict(), "quantized_enhancer.pth")

    # ==================================================
    # 新增的模型复杂度与推理速度分析
    # ==================================================
    print("\n正在分析原始模型...")
    original_report = generate_model_report(model)

    print("\n正在分析量化模型...")
    quantized_report = generate_model_report(quantized_model)

    # 示例推理
    test_images = ["test_image1.jpg", "test_image2.jpg"]
    enhanced_results = enhance_image(model, test_images, ["result1.jpg", "result2.jpg"], hdr_output=True)

    print("Enhancement complete!")