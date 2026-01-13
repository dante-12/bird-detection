# スクリプトで使用しているExifパラメータ一覧
## 画像の緯度・経度
    * GPS Latitude
    * GPS Longitude

## ドローンの高度
    * Relative Altitude - 不正確な場合があるため`--lake-alt`で補正が必要

## ドローンとカメラの向き
    * Flight Yaw Degree
    * Gimbal Yaw Degree
    * Flight Pitch Degree - 読み込んではいるが未使用。
    * Gimbal Pitch Degree - カメラは真下を向いていることを想定。-90度前後の値であることを期待。inspect-onlyモードでのみ仕様。
    * Gimbal Roll Degree - inspect-onlyモードでのみ、確認に使用。0度かnullであるべき。

## カメラの画角
    * Field Of View
