# GitHub 推送指南

## 步骤 1: 安装 Git

### Windows 用户
1. 访问 [Git for Windows](https://git-scm.com/download/win)
2. 下载并安装 Git
3. 安装完成后重启命令行

### 验证安装
```bash
git --version
```

## 步骤 2: 配置 Git

设置您的用户名和邮箱：
```bash
git config --global user.name "您的GitHub用户名"
git config --global user.email "您的邮箱地址"
```

## 步骤 3: 在 GitHub 上创建仓库

1. 访问 [GitHub](https://github.com)
2. 登录您的账户
3. 点击右上角的 "+" 号，选择 "New repository"
4. 填写仓库信息：
   - Repository name: `whisper-diarization-project`
   - Description: `基于 Whisper 和 pyannote.audio 的音频转写和说话人分离工具`
   - 选择 "Public" 或 "Private"
   - **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

## 步骤 4: 初始化本地仓库

在项目目录中运行以下命令：

```bash
# 初始化 Git 仓库
git init

# 添加所有文件到暂存区
git add .

# 创建第一个提交
git commit -m "Initial commit: Whisper 说话人分离转写工具

- 添加音频转写和说话人分离功能
- 支持多格式输出（文本和SRT字幕）
- 完善的错误处理和日志记录
- 灵活的配置文件支持
- 详细的文档和使用指南"

# 添加远程仓库（替换 YOUR_USERNAME 为您的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/whisper-diarization-project.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

## 步骤 5: 验证推送

1. 访问您的 GitHub 仓库页面
2. 确认所有文件都已成功上传
3. 检查 README.md 是否正确显示

## 项目文件结构

推送后，您的 GitHub 仓库应该包含以下文件：

```
whisper-diarization-project/
├── whisper_diarize_transcribe.py  # 主程序
├── config.json                    # 配置文件
├── requirements.txt               # 依赖管理
├── README.md                     # 项目文档
├── .gitignore                    # Git忽略文件
└── GITHUB_SETUP.md              # 本指南
```

## 常见问题

### 1. 如果遇到认证问题
```bash
# 使用个人访问令牌（推荐）
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/whisper-diarization-project.git
```

### 2. 如果需要更新代码
```bash
git add .
git commit -m "更新说明"
git push
```

### 3. 如果需要克隆到其他机器
```bash
git clone https://github.com/YOUR_USERNAME/whisper-diarization-project.git
```

## 下一步

1. 在 GitHub 上为您的项目添加标签和描述
2. 创建 Issues 来跟踪功能请求和 bug 报告
3. 邀请其他开发者参与项目
4. 考虑添加 GitHub Actions 来自动化测试和部署

## 项目特色

- 🎬 **视频转音频**：自动从视频文件提取音频
- 👥 **说话人分离**：识别不同的说话人并标记时间段
- 🧠 **智能转写**：使用 Whisper 进行高精度语音转文字
- 📝 **多格式输出**：生成文本文件和 SRT 字幕文件
- 📊 **统计信息**：提供详细的处理统计报告
- 🔧 **灵活配置**：支持配置文件自定义参数
- 📈 **进度显示**：实时显示处理进度
- 🛡️ **错误处理**：完善的错误处理和日志记录

祝您项目成功！🚀
