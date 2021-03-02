import wx                               # GUIを作成するためにwxPythonを使用
import cv2                              # カメラから取得した画像を処理するために使用
from GetCamera import Camera            # カメラに関するクラスを読み込み
from network.network import Network     # ネットワークに関するクラスを読み込み
from utils import *                     # その他必要な自作関数を読み込み

# GUIに関するクラス
class myFrame(wx.Frame):
    def __init__(self, parent, title, required_fps):
        # フレームの初期化
        # 画面サイズの変更を禁止(レイアウトが崩れることを防ぐため)
        wx.Frame.__init__(self, parent, title=title, size=(1085, 1000), style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        # カメラから取得した映像を表示する入れ物
        # 初期化時にNullBitmapを表示
        self.item = wx.StaticBitmap(self, bitmap=wx.NullBitmap, size=wx.Size(640, 480))
        # 撮影ボタンを押した際にカメラから取得した画像を表示する入れ物
        # 初期化時にNullBitmapを表示
        self.capture = wx.StaticBitmap(self, bitmap=wx.NullBitmap, size=wx.Size(640, 480))
        # 撮影ボタン
        self.photo_button = wx.Button(self, label="撮影")
        # モデルが予測したクラス名を表示する入れ物
        self.txt = wx.StaticText(self, -1, "", size=wx.Size(400, 30), style=wx.TE_CENTER)
        # テキストのフォントとフォントサイズを指定
        font = wx.Font(40, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        # テキストのフォントを先ほど指定したものにする
        self.txt.SetFont(font)
        # カメラ
        self.camera = Camera(0)
        # ネットワーク
        self.network = Network()
        # 映像の表示にタイマーを使用
        self.timer = wx.Timer(self)
        # ボタンクリックとOnButtonをバインド
        self.Bind(wx.EVT_BUTTON, self.OnButton)
        # timerによる時間経過とOnTimerをバインド
        self.Bind(wx.EVT_TIMER, self.OnTimer)

        # 画面を2×2に分割
        sizer = wx.GridSizer(2, 2, 2, 215)
        # カメラによる映像を配置
        sizer.Add(self.item)
        # ボタンを配置
        sizer.Add(self.photo_button, flag=wx.GROW)
        # 撮影ボタンを押した際にカメラから取得する画像を表示する部分を配置
        sizer.Add(self.capture)
        # テキストを配置
        sizer.Add(self.txt, flag=wx.ALIGN_CENTER)


        # タイマーによる実行間隔の指定
        # ms単位で指定するため1000÷(目標fps)の商(小数点以下切り捨て)を待ち時間とする
        wait = 1000 // required_fps
        # タイマーをスタートする
        self.timer.Start(wait)

        # 画面分割を設定
        # sizerによる分割指定及び配置
        self.SetSizer(sizer)

    # タイマーの消費ごとに実行される関数
    def OnTimer(self, event):
        # カメラ画像の取得
        ret, capture = self.camera.get_frame()
        # カメラから画像が取得されている場合のみ実行
        if ret:
            # openCVで読み込んだ画像はBGRの順になっているためRGBの順に変換
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
            # 画像をビットマップ形式に変換
            bmp = wx.Bitmap.FromBuffer(capture.shape[1], capture.shape[0], capture)
            # 画面に表示
            self.item.SetBitmap(bmp)
        else:
            print("NULL!!!")

    # ボタンが押された際に実行される関数
    def OnButton(self, event):
        # カメラ画像の取得
        ret, capture = self.camera.get_frame()
        # カメラから画像が取得されている場合のみ実行
        if ret:
            # 画面中央を正方形に切り取り
            cropped = crop_center(capture)
            # 切り取り後の画像を保存
            cv2.imwrite("./img/capture/cap.png", cropped)
            # 切り取り前の画像をBGRからRGBに変換
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
            # 画像をビットマップ形式に変換
            bmp = wx.Bitmap.FromBuffer(capture.shape[1], capture.shape[0], capture)
            # 画面に表示
            self.capture.SetBitmap(bmp)
            # 予測結果を得る
            pred = self.network.predict()
            # 予測結果を画面に表示
            self.txt.SetLabel(pred)
