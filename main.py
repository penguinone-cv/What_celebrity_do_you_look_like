import wx                       # GUIを作成するためにwxPythonを使用
import os                       # osの機能を使用
from Window import myFrame      # フレームクラスの読み込み

# main関数
def main():
    # GUIアプリケーションのラッパー
    app = wx.App()
    # フレーム定義
    frame = myFrame(None, "What celebrity do you look like?", 15)
    # フレームの表示
    frame.Show()

    # アプリの起動
    app.MainLoop()
    # 保存した画像を削除
    os.remove("./img/capture/cap.png")


if __name__=="__main__":
    main()
