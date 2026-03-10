#!/bin/bash

# 脚本：下载外部JS/CSS库到本地以支持离线运行
# 作者：Claude Assistant
# 用途：将WisdominDATA项目中的CDN依赖下载到本地

set -e  # 遇到错误时退出

echo "开始下载外部库到本地..."

# 创建必要的目录
mkdir -p static/js/vendor
mkdir -p static/css/vendor

# 1. Chart.js (图表库)
echo "下载 Chart.js..."
wget -O static/js/vendor/chart.min.js https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js

# 2. Plotly.js (图表库)
echo "下载 Plotly.js..."
wget -O static/js/vendor/plotly.min.js https://cdn.plot.ly/plotly-3.0.1.min.js

# 3. Prism.js (代码高亮)
echo "下载 Prism.js 相关文件..."
wget -O static/css/vendor/prism-okaidia.min.css https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css
wget -O static/js/vendor/prism.min.js https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js
wget -O static/js/vendor/prism-autoloader.min.js https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js

# 4. Moment.js (日期处理)
echo "下载 Moment.js..."
wget -O static/js/vendor/moment.min.js https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js
wget -O static/js/vendor/moment-zh-cn.min.js https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/locale/zh-cn.min.js

# 5. Sortable.js (拖拽排序)
echo "下载 Sortable.js..."
wget -O static/js/vendor/sortable.min.js https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js

# 6. Vis-Network (图谱可视化)
echo "下载 Vis-Network..."
wget -O static/js/vendor/vis-network.min.js https://unpkg.com/vis-network/standalone/umd/vis-network.min.js
wget -O static/css/vendor/vis-network.min.css https://unpkg.com/vis-network/styles/vis-network.min.css

# 7. Font Awesome (图标字体) - 需要特殊处理
echo "下载 Font Awesome CSS..."
wget -O static/css/vendor/font-awesome.min.css https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css

# 下载 Font Awesome 字体文件
echo "下载 Font Awesome 字体文件..."
mkdir -p static/webfonts
wget -O static/webfonts/fa-solid-900.woff2 https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-solid-900.woff2
wget -O static/webfonts/fa-regular-400.woff2 https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-regular-400.woff2
wget -O static/webfonts/fa-brands-400.woff2 https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-brands-400.woff2

# 修复 Font Awesome CSS 中的字体路径
echo "修复 Font Awesome CSS 字体路径..."
sed -i 's|https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/|../webfonts/|g' static/css/vendor/font-awesome.min.css

echo "所有库下载完成！"
echo ""
echo "下载的文件列表："
find static/js/vendor -type f | sort
find static/css/vendor -type f | sort
find static/webfonts -type f | sort

echo ""
echo "接下来运行 update_paths.py 脚本来更新HTML文件中的路径引用" 