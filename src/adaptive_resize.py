from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class PasteToCenterCanvas:
    """
    把输入图片粘贴到指定大小的白色画布中央。
    """

    def __init__(self, canvas_size=(512, 512), fill=(0, 0, 0)):
        self.canvas_size = canvas_size
        self.fill = fill

    def __call__(self, img):
        # img: PIL Image
        canvas = Image.new("RGB", self.canvas_size, self.fill)
        w, h = img.size
        canvas_w, canvas_h = self.canvas_size
        # 计算左上角坐标
        left = (canvas_w - w) // 2
        top = (canvas_h - h) // 2
        canvas.paste(img, (left, top))
        return canvas


class AdaptiveResizeMultipleOf:
    """
    一个自定义的 torchvision transform，它执行两个操作：
    1. 根据最大分辨率自适应地调整图像大小，保持宽高比。
    2. 确保最终的图像高和宽都是指定数字（例如16）的整数倍。
    """

    def __init__(self, max_size, multiple_of=16):
        """
        Args:
            max_size (int): 图像的最长边不应超过的最大尺寸。
            multiple_of (int): 最终输出尺寸需要是这个数字的倍数。
        """
        self.max_size = max_size
        self.multiple_of = multiple_of

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 需要被转换的图像。

        Returns:
            PIL Image: 调整大小后的图像。
        """
        # --- 步骤 1: 自适应调整大小 ---
        w, h = img.size

        # 计算初步的目标尺寸，保持宽高比
        if w > self.max_size or h > self.max_size:
            if w > h:
                target_w = self.max_size
                target_h = int(h * (self.max_size / w))
            else:
                target_h = self.max_size
                target_w = int(w * (self.max_size / h))
        else:
            target_w, target_h = w, h

        # --- 步骤 2: 调整为16的倍数 ---
        # 使用四舍五入的方式找到最接近的倍数
        final_w = int(round(target_w / self.multiple_of) * self.multiple_of)
        final_h = int(round(target_h / self.multiple_of) * self.multiple_of)

        # 防止尺寸变为0
        if final_w == 0:
            final_w = self.multiple_of
        if final_h == 0:
            final_h = self.multiple_of

        return T.functional.resize(
            img, (final_h, final_w), interpolation=InterpolationMode.LANCZOS
        )
