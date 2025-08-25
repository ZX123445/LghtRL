import os
import glob
import cv2
import numpy as np
from scipy import special
from skimage import color, filters
from collections import OrderedDict
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlindIQA')


class BlindImageQualityAnalyzer:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.feature_names = [
            'BRISQUE', 'NIQE', 'GMSD', 'Entropy', 'Contrast',
            'Sharpness', 'Colorfulness', 'NoiseLevel', 'Saturation'
        ]

    def analyze(self, img):
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray_img = img
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = {}
        results['BRISQUE'] = self.calculate_brisque(gray_img)
        results['NIQE'] = self.calculate_niqe(gray_img)
        results['GMSD'] = self.calculate_gmsd(gray_img)
        results['Entropy'] = self.calculate_entropy(gray_img)
        results['Contrast'] = self.calculate_contrast(gray_img)
        results['Sharpness'] = self.calculate_sharpness(gray_img)
        results['Colorfulness'] = self.calculate_colorfulness(rgb_img)
        results['NoiseLevel'] = self.calculate_noise_level(gray_img)
        results['Saturation'] = self.calculate_saturation(rgb_img)
        results['OverallScore'] = self.calculate_overall_score(results)
        return results

    def calculate_brisque(self, gray_img):
        try:
            mscn = self._calculate_mscn(gray_img)
            alpha, left_std, right_std = self._aggd_features(mscn)

            shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
            features = [alpha, left_std ** 2, right_std ** 2]

            for shift in shifts:
                shifted = np.roll(mscn, shift, axis=(0, 1))
                product = mscn * shifted
                alpha_p, left_std_p, right_std_p = self._aggd_features(product)
                features.extend([alpha_p, left_std_p ** 2, right_std_p ** 2])

            brisque_score = 100 - (np.mean(features) * 10)
            return np.clip(brisque_score, 0, 100)
        except Exception as e:
            logger.error(f"Error in BRISQUE calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_niqe(self, gray_img):
        try:
            mscn = self._calculate_mscn(gray_img)
            shape, loc, scale = self._ggd_features(mscn)

            shifts = [(0, 1), (1, 0)]
            features = [shape, loc, scale]

            for shift in shifts:
                shifted = np.roll(mscn, shift, axis=(0, 1))
                product = mscn * shifted
                alpha_p, left_std_p, right_std_p = self._aggd_features(product)
                features.extend([alpha_p, left_std_p, right_std_p])

            niqe_score = np.mean(features) * 20
            return np.clip(niqe_score, 0, 100)
        except Exception as e:
            logger.error(f"Error in NIQE calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_gmsd(self, gray_img):
        try:
            sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            mean_gm = np.mean(gradient_magnitude)
            if mean_gm < 1e-8:
                return 100.0

            similarity = (2 * gradient_magnitude * mean_gm) / (gradient_magnitude ** 2 + mean_gm ** 2 + 1e-8)
            gmsd = np.std(similarity)
            gmsd_score = 100 * (1 - gmsd)
            return np.clip(gmsd_score, 0, 100)
        except Exception as e:
            logger.error(f"Error in GMSD calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_entropy(self, gray_img):
        try:
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist /= hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            entropy_score = 100 * entropy / 8
            return np.clip(entropy_score, 0, 100)
        except Exception as e:
            logger.error(f"Error in Entropy calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_contrast(self, gray_img):
        try:
            min_val = np.min(gray_img)
            max_val = np.max(gray_img)
            if max_val - min_val < 1e-8:
                return 0.0
            contrast = (max_val - min_val) / (max_val + min_val + 1e-8)
            return 100 * contrast
        except Exception as e:
            logger.error(f"Error in Contrast calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_sharpness(self, gray_img):
        try:
            laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
            sharpness = np.var(laplacian)
            sharpness_score = 100 * min(sharpness / 1000, 1.0)
            return sharpness_score
        except Exception as e:
            logger.error(f"Error in Sharpness calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_colorfulness(self, rgb_img):
        try:
            r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            rg = r.astype(float) - g.astype(float)
            yb = 0.5 * (r.astype(float) + g.astype(float)) - b.astype(float)
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
            colorfulness_score = 100 * min(colorfulness / 100, 1.0)
            return colorfulness_score
        except Exception as e:
            logger.error(f"Error in Colorfulness calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_noise_level(self, gray_img):
        try:
            # 使用更简单的噪声估计方法
            denoised = cv2.medianBlur(gray_img, 3)
            # 修复了这里的拼写错误：astype 而不是 ast
            residual = gray_img.astype(np.float32) - denoised.astype(np.float32)
            noise_level = np.std(residual)
            noise_score = 100 * (1 - min(noise_level / 50, 1.0))
            return noise_score
        except Exception as e:
            logger.error(f"Error in NoiseLevel calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_saturation(self, rgb_img):
        try:
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv_img[:, :, 1])
            return saturation / 2.55  # 0-255 -> 0-100
        except Exception as e:
            logger.error(f"Error in Saturation calculation: {str(e)}")
            return 50.0  # 返回默认值

    def calculate_overall_score(self, metrics):
        weights = {
            'BRISQUE': 0.15, 'NIQE': 0.15, 'GMSD': 0.15,
            'Entropy': 0.10, 'Contrast': 0.10, 'Sharpness': 0.15,
            'Colorfulness': 0.05, 'NoiseLevel': 0.10, 'Saturation': 0.05
        }
        total_score = sum(metrics[key] * weight for key, weight in weights.items())
        return total_score

    def _calculate_mscn(self, img):
        try:
            blurred = cv2.GaussianBlur(img.astype(np.float64), (7, 7), 1.1666)
            mscn = (img - blurred) / (blurred + 1)
            return np.nan_to_num(mscn, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.error(f"Error in MSCN calculation: {str(e)}")
            return np.zeros_like(img, dtype=np.float64)

    def _aggd_features(self, coeffs):
        try:
            left = coeffs[coeffs < 0]
            right = coeffs[coeffs > 0]

            if len(left) == 0 or len(right) == 0:
                return 0.2, 0.2, 0.2

            left_std = np.std(left)
            right_std = np.std(right)
            gamma = left_std / right_std
            rho = np.mean(np.abs(coeffs)) ** 2 / np.mean(coeffs ** 2)
            alpha = np.sqrt(2 / (rho - 1)) if rho > 1 else 0.2
            return alpha, left_std, right_std
        except Exception as e:
            logger.error(f"Error in AGGD calculation: {str(e)}")
            return 0.2, 0.2, 0.2

    def _ggd_features(self, coeffs):
        try:
            variance = np.var(coeffs)
            if variance < 1e-8:
                return 0.5, 0.0, 0.5

            # 修复括号不匹配问题
            shape = np.sqrt(6 / (np.pi ** 2 * (np.log(2 / variance) + special.gammaln(1) - special.gammaln(3))))
            scale = np.sqrt(variance / np.exp(special.gammaln(1 / shape) - special.gammaln(3 / shape) + np.log(2)))
            loc = np.mean(coeffs)
            return shape, loc, scale
        except Exception as e:
            logger.error(f"Error in GGD calculation: {str(e)}")
            return 0.5, 0.0, 0.5


def imread(path):
    """更健壮的图像读取函数，处理各种格式和编码问题"""
    try:
        # 检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # 尝试标准读取
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # 如果标准读取失败，尝试其他方法
        if img is None:
            # 使用imdecode处理特殊编码
            with open(path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ValueError(f"OpenCV cannot read the image: {path}")

        # 处理不同通道数的图像
        if img.ndim == 2:
            return img
        elif img.shape[2] == 4:  # RGBA图
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 3:  # BGR图
            return img
        elif img.shape[2] == 1:  # 单通道
            return np.squeeze(img)
        else:
            # 对于其他通道数，转换为RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logger.error(f"Error reading image {path}: {str(e)}")
        return None


def analyze_directory(directory, use_gpu=False, verbose=True):
    extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']
    patterns = [os.path.join(directory, f'*.{ext}') for ext in extensions] + \
               [os.path.join(directory, f'*.{ext.upper()}') for ext in extensions]

    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(pattern, recursive=True))
    image_paths = natsorted(image_paths)

    if not image_paths:
        logger.error(f"No images found in directory: {directory}")
        return None

    logger.info(f"Found {len(image_paths)} images in: {directory}")

    analyzer = BlindImageQualityAnalyzer(use_gpu=use_gpu)
    results = []
    valid_images = 0
    failed_images = 0

    for path in tqdm(image_paths, desc="Processing images"):
        try:
            img = imread(path)
            if img is None:
                logger.warning(f"Skipping unreadable image: {os.path.basename(path)}")
                failed_images += 1
                continue

            # 确保图像是二维或三维数组
            if len(img.shape) not in [2, 3]:
                logger.warning(f"Invalid image dimensions for {os.path.basename(path)}: {img.shape}")
                failed_images += 1
                continue

            # 确保图像有有效的数据
            if img.size == 0 or np.max(img) == 0:
                logger.warning(f"Empty or invalid image data: {os.path.basename(path)}")
                failed_images += 1
                continue

            result = analyzer.analyze(img)
            result['filename'] = os.path.basename(path)
            results.append(result)
            valid_images += 1
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(path)}: {str(e)}")
            failed_images += 1

    if not results:
        logger.error("No images processed successfully")
        return None

    logger.info(f"Successfully processed {valid_images} images, failed on {failed_images} images")
    return {
        'images': results,
        'directory': directory,
        'num_images': valid_images
    }


def save_results(results, output_dir):
    if not results:
        logger.warning("No results to save")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results to: {output_dir}")

    # Save detailed results
    csv_path = os.path.join(output_dir, 'quality_scores.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(
                'filename,OverallScore,BRISQUE,NIQE,GMSD,Entropy,Contrast,Sharpness,Colorfulness,NoiseLevel,Saturation\n')
            for res in results['images']:
                f.write(f"{res['filename']},{res['OverallScore']:.4f},{res['BRISQUE']:.4f},{res['NIQE']:.4f},")
                f.write(f"{res['GMSD']:.4f},{res['Entropy']:.4f},{res['Contrast']:.4f},{res['Sharpness']:.4f},")
                f.write(f"{res['Colorfulness']:.4f},{res['NoiseLevel']:.4f},{res['Saturation']:.4f}\n")
        logger.info(f"Saved detailed results to: {csv_path}")
    except Exception as e:
        logger.error(f"Error saving CSV results: {str(e)}")

    # Calculate statistics
    metrics = ['OverallScore', 'BRISQUE', 'NIQE', 'GMSD', 'Entropy',
               'Contrast', 'Sharpness', 'Colorfulness', 'NoiseLevel', 'Saturation']
    stats = {metric: [] for metric in metrics}

    for res in results['images']:
        for metric in metrics:
            stats[metric].append(res[metric])

    # Save summary report
    report_path = os.path.join(output_dir, 'summary_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Image Quality Analysis Report\n")
            f.write(f"Directory: {results['directory']}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Processed: {results['num_images']}\n\n")

            f.write("Average Scores:\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Metric':<15} {'Average':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}\n")
            f.write("-" * 60 + "\n")

            for metric in metrics:
                values = stats[metric]
                if values:  # 确保列表不为空
                    f.write(f"{metric:<15} {np.mean(values):<10.4f} {np.min(values):<10.4f} ")
                    f.write(f"{np.max(values):<10.4f} {np.std(values):<10.4f}\n")
                else:
                    f.write(f"{metric:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n")

        logger.info(f"Saved summary report to: {report_path}")
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")

    # 创建可视化图表
    try:
        # 质量分数分布图
        plt.figure(figsize=(12, 6))
        scores = [res['OverallScore'] for res in results['images']]
        plt.bar(range(len(scores)), scores)
        plt.title('Image Quality Scores')
        plt.xlabel('Image Index')
        plt.ylabel('Overall Score')
        plt.tight_layout()
        quality_scores_path = os.path.join(output_dir, 'quality_scores.png')
        plt.savefig(quality_scores_path)
        plt.close()
        logger.info(f"Saved quality scores plot to: {quality_scores_path}")

        # 指标相关性热力图
        plt.figure(figsize=(10, 8))
        metrics_subset = ['BRISQUE', 'NIQE', 'GMSD', 'Entropy', 'Contrast', 'Sharpness']
        data = []
        for res in results['images']:
            row = [res[m] for m in metrics_subset]
            data.append(row)

        if data:  # 确保数据不为空
            corr_matrix = np.corrcoef(np.array(data).T)

            sns.heatmap(corr_matrix, annot=True, xticklabels=metrics_subset,
                        yticklabels=metrics_subset, cmap='coolwarm')
            plt.title('Quality Metric Correlations')
            plt.tight_layout()
            corr_matrix_path = os.path.join(output_dir, 'correlation_matrix.png')
            plt.savefig(corr_matrix_path)
            plt.close()
            logger.info(f"Saved correlation matrix to: {corr_matrix_path}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")

    return csv_path, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blind Image Quality Assessment')
    parser.add_argument('--dir', type=str, default='H:\研究生论文/10_LOL/1/over_Net_testmap/test_results/test_results',
                        help='Directory containing images to analyze')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU acceleration if available')
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # 设置日志级别
    logging.basicConfig(level=getattr(logging, args.log),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info(f"Starting analysis of: {args.dir}")
    results = analyze_directory(args.dir, args.use_gpu)

    if results:
        logger.info(f"\nSaving results to: {args.output}")
        save_results(results, args.output)
        logger.info("Analysis completed successfully!")
    else:
        logger.error("Analysis failed. No results to save.")