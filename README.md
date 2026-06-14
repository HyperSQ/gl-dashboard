# GL Network Analysis Dashboard

## 部署 Render / 其他平台

### 文件清单
- `Procfile` → Render 启动命令
- `requirements.txt` → Python 依赖
- `data/*.pkl` → 预计算数据（不需 HDF5）

### Render 部署步骤
1. 将 `webapp/` 目录初始化为 Git 仓库并推送到 GitHub
2. 在 Render 创建新 Web Service，连接仓库
3. Build Command: `pip install -r requirements.txt`
4. Start Command: Render 自动读取 Procfile
5. 等待构建完成（首次约 5-10 分钟，数据文件较大）

### 注意
- 数据 pkl 文件总计约 200MB，Git push 较慢
- 如果 GitHub 拒绝 >100MB 的文件，需用 Git LFS

### 本地运行
```bash
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:8050
```
