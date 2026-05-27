# 数据集目录（不包含在 Git 仓库中）

`release/train` 与 `release/test` 为本地/赛方数据，含 `video.mp4`、`action.txt`、`instruction.txt` 等。**视频与轨迹大文件不会上传到 GitHub**。

可使用 `tools/prepare_release_dataset.py` 从数据根目录生成 manifest，供批量 TI2V 与世界模型评测使用。

目录结构示例：

```
release/
  train/<sample_id>/video.mp4, action.txt, ...
  test/<sample_id>/instruction.txt, ...
```
