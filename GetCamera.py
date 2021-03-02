import cv2      # カメラを使用するためにopenCVを使用

# カメラに関するクラス
# 参考：https://watlab-blog.com/2019/09/22/webcamera-realtime/
class Camera:
    def __init__(self, index):
        # カメラの定義
        self.camera = cv2.VideoCapture(index)
        # カメラのビデオコーデックをH264に指定
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))

    # カメラから画像を取得する関数
    def get_frame(self):
        # retはreadが正常にフレーム取得に成功したかを保存するフラグ
        ret, frame = self.camera.read()
        return ret, frame
