import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
from Network import LowLightEnhancer, LowLightDataset  # 

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Low Light Image Enhancement Testing')
    parser.add_argument('--model_path', type=str, default='best_enhancer.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--test_dir', type=str, default='test_data',
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='enhanced_results',
                        help='Directory to save enhanced images')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save attention maps and visualizations')
    parser.add_argument('--hdr_output', action='store_true',
                        help='Generate HDR version of enhanced images')
    parser.add_argument('--quantized', action='store_true',
                        help='Use quantized model for inference')
    return parser.parse_args()


def load_model(model_path, quantized=False):
    """加载训练好的模型"""
    model = LowLightEnhancer().to(device)

    # 加载模型权重
    if quantized:
        # 加载量化模型
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
        )
        # 量化模型需要特殊处理
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded quantized model")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded full precision model")

    model.eval()
    return model


def create_test_dataset(test_dir):
    """创建测试数据集"""
    # 如果没有提供测试集目录，使用默认验证集
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Using validation dataset instead.")
        return LowLightDataset(train=False)

    # 自定义测试数据集（仅包含低光照图像）
    class TestDataset(LowLightDataset):
        def __init__(self, test_dir):
            self.low_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.eval_transform = ToTensor()

        def __len__(self):
            return len(self.low_paths)

        def __getitem__(self, idx):
            low_img = Image.open(self.low_paths[idx]).convert('RGB')
            low_img = self.eval_transform(low_img)
            return low_img, os.path.basename(self.low_paths[idx])

    return TestDataset(test_dir)


def enhance_batch(model, low_imgs):
    """增强一批图像"""
    with torch.no_grad():
        enhanced, illum_map, color_corrected, color_attn = model(low_imgs)
    return enhanced, illum_map, color_corrected, color_attn


def save_results(enhanced, filenames, output_dir, hdr_output=False):
    """保存增强结果"""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(enhanced.shape[0]):
        img = enhanced[i].cpu().clamp(0, 1)
        filename = filenames[i]

        # 保存增强图像
        save_path = os.path.join(output_dir, f"enhanced_{filename}")
        save_image(img, save_path)

        # 保存HDR版本
        if hdr_output:
            hdr_path = os.path.join(output_dir, f"hdr_{os.path.splitext(filename)[0]}.hdr")
            hdr_img = img.permute(1, 2, 0).numpy() * 6.0  # 扩展动态范围
            cv2.imwrite(hdr_path, cv2.cvtColor(hdr_img, cv2.COLOR_RGB2BGR))


def save_attention_maps(illum_map, color_corrected, color_attn, filenames, output_dir):
    """保存注意力图和中间结果"""
    attn_dir = os.path.join(output_dir, "attention_maps")
    os.makedirs(attn_dir, exist_ok=True)

    for i in range(illum_map.shape[0]):
        filename = filenames[i]
        base_name = os.path.splitext(filename)[0]

        # 保存光照图
        illum = illum_map[i].cpu().clamp(0, 1)
        save_image(illum, os.path.join(attn_dir, f"{base_name}_illum_map.png"))

        # 保存颜色校正图
        color_corr = color_corrected[i].cpu().clamp(0, 1)
        save_image(color_corr, os.path.join(attn_dir, f"{base_name}_color_correction.png"))

        # 保存颜色注意力图
        color_att = color_attn[i].cpu().clamp(0, 1)
        save_image(color_att, os.path.join(attn_dir, f"{base_name}_color_attention.png"))


def visualize_comparison(low_img, enhanced, filename, output_dir):
    """创建并保存比较可视化图"""
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    # 转换为numpy
    low_np = low_img.cpu().numpy().transpose(1, 2, 0)
    enhanced_np = enhanced.cpu().numpy().transpose(1, 2, 0)

    # 创建比较图
    plt.figure(figsize=(15, 8))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(low_np, 0, 1))
    plt.title("Low Light Input")
    plt.axis('off')

    # 增强结果
    plt.subplot(2, 3, 2)
    plt.imshow(np.clip(enhanced_np, 0, 1))
    plt.title("Enhanced Result")
    plt.axis('off')

    # 颜色直方图比较
    plt.subplot(2, 3, 3)
    for j, color in enumerate(['r', 'g', 'b']):
        plt.hist(enhanced_np[:, :, j].flatten(), bins=50, alpha=0.5, color=color, label=f"Enhanced {color.upper()}")
        plt.hist(low_np[:, :, j].flatten(), bins=50, alpha=0.3, color=color, label=f"Low {color.upper()}",
                 histtype='step', linewidth=2)
    plt.title("Color Histograms")
    plt.legend()

    # 转换为HSV空间
    low_hsv = cv2.cvtColor(np.clip(low_np, 0, 1), cv2.COLOR_RGB2HSV)
    enhanced_hsv = cv2.cvtColor(np.clip(enhanced_np, 0, 1), cv2.COLOR_RGB2HSV)

    # 色相通道
    plt.subplot(2, 3, 4)
    plt.imshow(low_hsv[:, :, 0], cmap='hsv')
    plt.title("Low Light Hue")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(enhanced_hsv[:, :, 0], cmap='hsv')
    plt.title("Enhanced Hue")
    plt.colorbar()

    # 饱和度分布
    plt.subplot(2, 3, 6)
    plt.hist(low_hsv[:, :, 1].flatten(), bins=50, alpha=0.5, label='Low')
    plt.hist(enhanced_hsv[:, :, 1].flatten(), bins=50, alpha=0.5, label='Enhanced')
    plt.title("Saturation Distribution")
    plt.legend()

    # 保存图像
    plt.savefig(os.path.join(comp_dir, f"comparison_{os.path.splitext(filename)[0]}.png"), bbox_inches='tight')
    plt.close()


def measure_inference_speed(model, input_size=(1, 3, 256, 256), repetitions=100):
    """测量模型推理速度"""
    print("Measuring inference speed...")

    # 创建虚拟输入
    dummy_input = torch.randn(input_size).to(device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # 测量推理时间
    timings = []
    for _ in range(repetitions):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.time()
        timings.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 计算统计数据
    avg_time = sum(timings) / repetitions
    min_time = min(timings)
    max_time = max(timings)

    print(f"Inference speed results ({input_size[2]}x{input_size[3]} images):")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.2f}")

    return avg_time


def main():
    args = parse_arguments()

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.quantized)

    print(f"Loading test dataset from: {args.test_dir}")
    test_dataset = create_test_dataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Results will be saved to: {args.output_dir}")

    # 测量推理速度
    measure_inference_speed(model)

    # 处理所有测试图像
    total_time = 0
    num_images = 0

    for batch in test_loader:
        if isinstance(test_dataset, LowLightDataset) and not hasattr(test_dataset, 'low_paths'):
            # 验证集模式
            low_imgs, high_imgs = batch
            filenames = [f"image_{i + num_images}.png" for i in range(low_imgs.shape[0])]
        else:
            # 自定义测试集模式
            low_imgs, filenames = batch

        low_imgs = low_imgs.to(device)

        # 增强图像
        start_time = time.time()
        enhanced, illum_map, color_corrected, color_attn = enhance_batch(model, low_imgs)
        batch_time = time.time() - start_time

        # 保存结果
        save_results(enhanced, filenames, args.output_dir, args.hdr_output)

        # 保存注意力图和可视化
        if args.save_visualizations:
            save_attention_maps(illum_map, color_corrected, color_attn, filenames, args.output_dir)

            # 为每个图像创建比较图
            for i in range(low_imgs.shape[0]):
                visualize_comparison(low_imgs[i], enhanced[i], filenames[i], args.output_dir)

        # 更新计时统计
        total_time += batch_time
        num_images += low_imgs.shape[0]

        print(f"Processed {num_images} images | Last batch: {batch_time * 1000:.1f} ms")

    # 打印性能总结
    if num_images > 0:
        avg_time = total_time / num_images * 1000  # 每张图像平均毫秒
        fps = num_images / total_time
        print("\n" + "=" * 50)
        print(f"Total images processed: {num_images}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time:.1f} ms")
        print(f"Average FPS: {fps:.2f}")
        print("=" * 50)

    print("Enhancement complete!")


if __name__ == "__main__":

    main()
