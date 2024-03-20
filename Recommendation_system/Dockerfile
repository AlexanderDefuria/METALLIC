# 使用官方Python运行时作为父镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器中的/app
COPY . /app

# 安装requirements.txt中的所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 使端口8080可用于外界访问
EXPOSE 8080

# 在容器启动时运行Flask应用
CMD ["python", "run_waitress.py"]
