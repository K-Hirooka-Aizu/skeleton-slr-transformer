import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import RandomAffine, InterpolationMode
from typing import Tuple, Union, List, Optional, Dict, Any



class ResizeVideo(nn.Module):
    """
    (C, T, H, W) 形式のビデオテンソルをリサイズするクラス。
    torchvision.transforms.Compose で利用可能です。
    """

    def __init__(
        self, 
        size: Union[int, Tuple[int, int]], 
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True
    ):
        """
        Args:
            size (sequence or int): 出力サイズ。
                (h, w) のタプルの場合はそのサイズに強制変換（アスペクト比無視）。
                int の場合は、短い辺がそのサイズになるようにアスペクト比を維持してリサイズ。
            interpolation (InterpolationMode): 補間方法（デフォルト: BILINEAR）
            antialias (bool): アンチエイリアス処理を行うかどうか（デフォルト: True）
        """
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid (Tensor): (C, T, H, W) または (T, H, W) などの形式。
                          最後の2次元が (H, W) である必要があります。
        Returns:
            Tensor: リサイズされたビデオテンソル (C, T, H_new, W_new)
        """
        # F.resize は入力の最後の2次元 (H, W) に対して作用し、
        # それより前の次元 (C, T) はすべて保持されます。
        return F.resize(
            vid, 
            self.size, 
            interpolation=self.interpolation, 
            antialias=self.antialias
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation})"
    
class CenterCropVideo(nn.Module):
    """
    (C, T, H, W) 形式のビデオテンソルを中心から切り抜くクラス。
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super().__init__()
        # size が int なら (size, size) に、そうでなければそのまま使用
        self.size = (size, size) if isinstance(size, int) else size

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid (Tensor): (C, T, H, W)
        Returns:
            Tensor: Center Cropped video (C, T, size[0], size[1])
        """
        # CenterCropは「中心」という固定位置なので、
        # (C, T, H, W) をそのまま渡しても全フレーム同じ位置で切られます。
        return F.center_crop(vid, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    
class NormalizeVideo(nn.Module):
    """
    (C, T, H, W) 形式のビデオテンソルを正規化するクラス。
    input[channel] = (input[channel] - mean[channel]) / std[channel]
    """

    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        """
        Args:
            mean (sequence): 各チャンネルの平均値 (例: [0.485, 0.456, 0.406])
            std (sequence): 各チャンネルの標準偏差 (例: [0.229, 0.224, 0.225])
            inplace (bool): インプレース操作を行うかどうか
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid (Tensor): 正規化するビデオテンソル。形状は (C, T, H, W)。
                          値の範囲は通常 [0, 1] であることを想定しています。
        Returns:
            Tensor: 正規化されたビデオ (C, T, H, W)
        """
        # Tensorへの変換とデバイス同期
        mean = torch.as_tensor(self.mean, dtype=vid.dtype, device=vid.device)
        std = torch.as_tensor(self.std, dtype=vid.dtype, device=vid.device)

        # ブロードキャスト用に形状を (C, 1, 1, 1) に変換
        # これにより、時間(T)、高さ(H)、幅(W)の全画素に対して一括計算できます
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1, 1)

        # 正規化計算 (vid - mean) / std
        if self.inplace:
            vid.sub_(mean).div_(std)
            return vid
        else:
            return (vid - mean) / std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomCropVideo(nn.Module):
    """
    (C, T, H, W) 形式のビデオテンソルをランダムに切り抜くクラス。
    【重要】 時間軸 (T) 全体に対して「同じ位置」を切り抜きます（Temporal Consistency）。
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid (Tensor): (C, T, H, W)
        Returns:
            Tensor: Random Cropped video (C, T, size[0], size[1])
        """
        # 1. 切り抜く座標 (top, left, height, width) を決定する
        #    vidの最後の2次元 (H, W) を見て、ランダムな座標を算出します。
        i, j, h, w = transforms.RandomCrop.get_params(vid, output_size=self.size)

        # 2. 決定した座標を使って、全フレームを一括で切り抜く
        #    F.crop は (..., H, W) に対して作用するため、
        #    (C, T) 次元は保持されたまま、指定した矩形で切り抜かれます。
        return F.crop(vid, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    
class RandomAffineVideo(RandomAffine):
    """
    (C, T, H, W) 形式のビデオテンソルにランダムなアフィン変換を適用するクラス。
    
    【重要】
    時間的な整合性を保つため、1つの動画内の全フレームに対して
    「全く同じ」角度・移動・スケール・シアーを適用します。
    """

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Union[int, float, List[float]] = 0,
        center: Optional[List[int]] = None
    ):
        # 親クラス(RandomAffine)の初期化を呼び出して、パラメータのパースを任せる
        super().__init__(degrees, translate, scale, shear, interpolation, fill, center)

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid (Tensor): (C, T, H, W)
        Returns:
            Tensor: Affine Transformed Video (C, T, H, W)
        """
        # 1. 画像サイズ (H, W) を取得
        #    vid.shape は (C, T, H, W) なので、最後の2つを取得
        img_size = [vid.shape[-2], vid.shape[-1]]

        # 2. ランダムなパラメータを「1回だけ」生成する
        #    get_params は staticmethod なので、クラス経由で呼び出します。
        #    これにより、全フレームで共通の params (angle, translations, scale, shear) が決まります。
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        
        angle, translations, scale, shear = ret

        # 3. 生成したパラメータを適用する
        #    F.affine は (..., H, W) の形式の入力に対応しており、
        #    先頭の次元 (C, T) はバッチのように扱われて、一括で同じ変換が適用されます。
        return F.affine(
            vid,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center
        )

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(degrees={self.degrees}'
        if self.translate is not None:
            s += f', translate={self.translate}'
        if self.scale is not None:
            s += f', scale={self.scale}'
        if self.shear is not None:
            s += f', shear={self.shear}'
        if self.interpolation != InterpolationMode.NEAREST:
            s += f', interpolation={self.interpolation.value}'
        if self.fill != 0:
            s += f', fill={self.fill}'
        if self.center is not None:
            s += f', center={self.center}'
        s += ')'
        return s
    
TRANSFORM_REGISTRY = {
    "ResizeVideo": ResizeVideo,
    "CenterCropVideo": CenterCropVideo,
    "RandomCropVideo": RandomCropVideo,
    "RandomAffineVideo": RandomAffineVideo,
    "NormalizeVideo": NormalizeVideo,
}

def build_transforms_from_config(config_list: List[Dict[str, Any]]) -> transforms.Compose:
    """
    設定リストから transform パイプラインを動的に構築する関数。

    Args:
        config_list (List[Dict]): 以下のような辞書のリスト
            [
                {"name": "ToFloatTensorVideo"},
                {"name": "ResizeVideo", "args": {"size": 256}},
                {"name": "RandomCropVideo", "args": {"size": 224}},
                ...
            ]
    
    Returns:
        transforms.Compose: 構築されたパイプライン
    """
    transform_instances = []

    for config in config_list:
        name = config.get("name")
        args = config.get("args", {}) # 引数がない場合は空辞書

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' is not found in registry. Available: {list(TRANSFORM_REGISTRY.keys())}")

        # クラスを取得
        transform_class = TRANSFORM_REGISTRY[name]

        try:
            # 引数を展開してインスタンス化 (**args)
            instance = transform_class(**args)
            transform_instances.append(instance)
            # print(f"Built: {name} with args: {args}") # デバッグ用
        except TypeError as e:
            raise TypeError(f"Failed to instantiate {name}. Check arguments: {args}. Error: {e}")

    return transforms.Compose(transform_instances)

if __name__=="__main__":
    import torchvision.transforms as transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # 設定：224x224 に強制リサイズし、0-1に正規化するパイプライン
    train_transform = transforms.Compose([        
        ResizeVideo(size=256),
        RandomAffineVideo(
            degrees=15, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1), 
            shear=10
        ),
        RandomCropVideo(size=224),
        NormalizeVideo(mean=imagenet_mean, std=imagenet_std)
    ])

    # dummy test
    dummy_video = torch.randn(3, 64, 1080, 1920) # (C, T, H, W)
    out = train_transform(dummy_video)

    print(out.shape) # torch.Size([3, 64, 224, 224])
