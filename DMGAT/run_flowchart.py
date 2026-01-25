# run_flowchart.py (最终完美版)

# 1. 从库中导入你找到的、更方便的函数
from py2flowchart import pyfile2flowchart

# 2. 定义输入和输出文件名
input_py_file = 'main.py'
output_html_file = 'main_flowchart.html'

print(f"准备将 '{input_py_file}' 转换为HTML流程图...")
print(f"输出文件将会是: '{output_html_file}'")

try:
    # 3. 直接调用函数，一步完成所有转换工作
    pyfile2flowchart(input_py_file, output_html_file)
    
    print("\n转换成功！🎉")
    print(f"请在项目文件夹中查找并用浏览器打开 '{output_html_file}' 文件查看结果。")

except FileNotFoundError:
    print(f"\n错误：找不到输入文件 '{input_py_file}'。请确保它和本脚本在同一个文件夹里。")
except Exception as e:
    print(f"\n发生了一个意外错误: {e}")