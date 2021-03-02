import numpy as np                                                  # torch tensorからndarrayに変換してargumentをint型として扱う
import torch                                                        # Deep Learningのベースライブラリとして使用
from torch import utils                                             # pytorch内のDataLoaderを使用
from torchvision import datasets, transforms                        # ImageFolderと前処理の設定に使用
from network.model import *                                         # モデル定義
from pytorch_metric_learning import losses                          # 損失関数はpytorch metric learningから使用
from pytorch_metric_learning.utils.inference import LogitGetter     # pytorch metric learning内のLogitGetterを使用

# ネットワーク部分
# 推論を行う関数を定義している
class Network:
    def __init__(self):
        # GPUが利用可能であればGPUを使用，利用可能でなければCPUを使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # モデルを定義しself.deviceのメモリに送る
        self.model = PretrainedResNet(512).to(self.device)
        # 学習済み重みの読み込み
        self.model.load_state_dict(torch.load("./network/model/arcface_SGD"))
        # 損失関数を定義しself.deviceのメモリに送る
        self.loss = losses.ArcFaceLoss(margin=14.3, scale=64, num_classes=40, embedding_size=512).to(self.device)
        # 損失関数の学習済み重みの読み込み(ArcFaceLossの内部には学習可能な部分があるため，その部分の重みを読み込む)
        self.loss.load_state_dict(torch.load("./network/model/arcface_SGD_loss"))

        # モデルを評価モード(重みを固定するモード)に変更
        self.model.eval()

        # 画像の前処理方法を指定
        # 144×144にリサイズ
        # Torch Tensor(pytorchで使用可能な配列)に変換
        # RGB各チャネルの画素値を平均0.5，標準偏差0.5の正規分布に従うように変換(およそ0~1の範囲になる)
        self.transform = transforms.Compose([transforms.Resize((144, 144)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # モデルの最終出力は損失関数から得る必要がある(モデルの出力は512次元のベクトル，得たい出力は40次元のベクトル)
        self.logit_getter = LogitGetter(self.loss)
        # argumentとクラス名の対応表
        # モデル出力は入力が各クラスである確率として得られるため，対応する配列を作っておく必要がある
        self.pred_class = ("新垣結衣", "新木優子", "新田真剣佑", "綾瀬はるか", "ディーンフジオカ", "深田恭子", "浜辺美波",
                        "橋本環奈", "比嘉愛未", "平野紫耀", "広瀬すず", "本田翼", "井川遥", "石原さとみ", "桐谷美玲",
                        "北川景子", "松坂桃李", "三浦春馬", "向井理", "長澤まさみ", "中川大志", "中条あやみ", "中村倫也",
                        "小栗旬", "岡田将生", "坂口健太郎", "佐々木希","佐藤健", "沢尻エリカ", "瀬戸康史", "白石麻衣",
                        "柴咲コウ", "菅田将暉", "竹野内豊", "玉木宏", "山本美月", "山下智久", "山崎賢人", "横浜流星", "吉沢亮")

    # モデルを用いた推論を行う関数
    def predict(self):
        # 保存した画像をモデルに入力しやすい形に変換する
        # 1. 保存した画像の読み込みとself.transformによる前処理を施す
        # 2. pythonのイテレータ(アクセスする度に配列のデータを順に呼び出すデータ型)に変換
        data = datasets.ImageFolder(root="./img", transform=self.transform)
        loader = utils.data.DataLoader(data, batch_size=1, shuffle=False)

        # 推論
        # for文によるループで実装しているが画像は1枚のみ保存されるため1周で終了する
        for input, _ in loader:
            # 入力データをself.deviceのメモリに送る
            input = input.to(self.device)
            # モデルの最終出力を得る
            logits = self.logit_getter(self.model(input))
            # 出力が最大値となるargumentをCPUメモリに送り，ndarrayに変換
            pred_arg = np.asarray(logits.argmax(dim=1).to("cpu"))
            
            # モデルが予測したクラス名を返す
            return self.pred_class[pred_arg[0]]
