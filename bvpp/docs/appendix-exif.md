# スクリプトで使用しているExifパラメータ一覧
## 画像の緯度・経度
    * GPS Latitude
    * GPS Longitude

## ドローンの高度
    * Relative Altitude - 不正確な場合が多いため`--alt-correction`で補正が必要

## ドローンとカメラの向き
    * Flight Yaw Degree
    * Gimbal Yaw Degree
    * Flight Pitch Degree - ドローンの水平からのズレ。ほぼ0（水平）であるべき。0でない場合、カメラが直下を向いていないことを意味する。参照・表示のみで、描画には未使用。
    * Gimbal Pitch Degree - カメラは真下を向いていることを想定。-90度前後の値であることを期待。参照・表示のみで、描画には未使用。
    * Gimbal Roll Degree - inspect-onlyモードでのみ、確認に使用。0度かnullであるべき。参照・表示のみで、描画には未使用。

## カメラの画角
    * Field Of View - Exifのこの値が合っているのかどうか、正直疑問です
