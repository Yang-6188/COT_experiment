#!/bin/bash
echo "正在安装中文字体..."
sudo apt-get update -qq
sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei

echo "清除 matplotlib 缓存..."
rm -rf ~/.cache/matplotlib

echo "验证字体安装..."
python3 << 'EOF'
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'WenQuanYi' in f.name]
if fonts:
    print(f"✓ 成功安装字体: {fonts}")
else:
    print("✗ 字体安装可能失败")
EOF

echo "完成！请重新运行您的脚本。"
