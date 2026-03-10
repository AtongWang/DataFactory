#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本：更新HTML文件中的CDN路径为本地路径
作者：Claude Assistant
用途：将WisdominDATA项目中的CDN链接替换为本地文件路径
"""

import os
import re
import glob
from pathlib import Path

def main():
    print("开始更新HTML文件中的CDN路径...")
    
    # 定义CDN到本地路径的映射
    cdn_mappings = {
        # Chart.js
        'https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js': 
            "{{ url_for('static', filename='js/vendor/chart.min.js') }}",
        
        # Plotly.js
        'https://cdn.plot.ly/plotly-3.0.1.min.js': 
            "{{ url_for('static', filename='js/vendor/plotly.min.js') }}",
        
        # Prism.js CSS
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css': 
            "{{ url_for('static', filename='css/vendor/prism-okaidia.min.css') }}",
        
        # Prism.js JS
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js': 
            "{{ url_for('static', filename='js/vendor/prism.min.js') }}",
        
        # Prism.js Autoloader
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js': 
            "{{ url_for('static', filename='js/vendor/prism-autoloader.min.js') }}",
        
        # Moment.js
        'https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js': 
            "{{ url_for('static', filename='js/vendor/moment.min.js') }}",
        
        # Moment.js 中文语言包
        'https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/locale/zh-cn.min.js': 
            "{{ url_for('static', filename='js/vendor/moment-zh-cn.min.js') }}",
        
        # Sortable.js
        'https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js': 
            "{{ url_for('static', filename='js/vendor/sortable.min.js') }}",
        
        # Vis-Network JS
        'https://unpkg.com/vis-network/standalone/umd/vis-network.min.js': 
            "{{ url_for('static', filename='js/vendor/vis-network.min.js') }}",
        
        # Vis-Network CSS
        'https://unpkg.com/vis-network/styles/vis-network.min.css': 
            "{{ url_for('static', filename='css/vendor/vis-network.min.css') }}",
        
        # Font Awesome - 只处理完整的URL，不处理有integrity属性的
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css': 
            "{{ url_for('static', filename='css/vendor/font-awesome.min.css') }}"
    }
    
    # 查找所有HTML文件
    html_files = glob.glob('templates/*.html')
    
    updated_files = []
    
    for html_file in html_files:
        print(f"处理文件: {html_file}")
        
        # 读取文件内容
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 对每个CDN映射进行替换
        for cdn_url, local_path in cdn_mappings.items():
            if cdn_url in content:
                print(f"  替换: {cdn_url}")
                print(f"  替换为: {local_path}")
                content = content.replace(cdn_url, local_path)
        
        # 特殊处理Font Awesome的integrity属性
        # 匹配包含integrity属性的Font Awesome链接
        fa_pattern = r'<link rel="stylesheet" href="https://cdnjs\.cloudflare\.com/ajax/libs/font-awesome/6\.0\.0/css/all\.min\.css"[^>]*integrity="[^"]*"[^>]*crossorigin="[^"]*"[^>]*>'
        fa_replacement = '<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/vendor/font-awesome.min.css\') }}">'
        
        if re.search(fa_pattern, content):
            print(f"  特殊替换Font Awesome链接（包含integrity）")
            content = re.sub(fa_pattern, fa_replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(content)
            updated_files.append(html_file)
            print(f"  ✓ 文件已更新")
        else:
            print(f"  - 无需更新")
        
        print()
    
    print("="*60)
    print("路径更新完成！")
    print(f"共更新了 {len(updated_files)} 个文件:")
    for file in updated_files:
        print(f"  - {file}")
    
    print("\n请确保已运行 download_offline_libs.sh 脚本下载了所需的库文件。")
    print("现在您的应用应该可以离线运行了！")

if __name__ == "__main__":
    main() 