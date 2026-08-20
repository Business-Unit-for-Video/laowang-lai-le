# Laowang Lai Le

“老王来了”YouTube 频道的独立转录仓库，默认来源为：

- `https://www.youtube.com/@dlw2023`

## 目录

- `scripts/transcribe_youtube_channel.py`：频道发现、逐视频下载、转录、校验和状态维护
- `scripts/transcription_integrity.py`：音频与转录完整性检查
- `youtube_channels/老王来了/`：带时间戳文本、纯文本及 manifest
- `state_youtube/老王来了/`：队列、完成/失败状态、错误记录和进度
- `.github/workflows/transcribe_youtube_channel.yml`：手动或每 6 小时扫描，每次处理一个视频并按 `continue.flag` 串行续跑

## GitHub Secret

在本仓库配置 `YOUTUBE_COOKIES`。Cookie 只通过 Actions Secret 注入，不写入仓库。

## 兼容迁移说明

本仓库从 `Business-Unit-for-Video/Video2Text` 独立出来，保留了原有 `youtube_channels/老王来了/` 和 `state_youtube/老王来了/` 路径以及历史状态，避免重复处理。原 `Video2Text` 仓库暂不清理，待本仓库验证后再单独处理。

## 运行原则

- 一个 Run 只处理一个视频。
- 每 6 小时扫描新内容；存在待处理项时由 `continue.flag` 串行触发下一次 Run。
- 默认不包含会员内容。
- 不打印 Cookie，不把媒体临时文件提交到 Git。
- 请确保源内容具有相应的授权、许可或其他合法使用依据。
