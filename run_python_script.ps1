#定义参数
param([int]$Index)
$Index
# 切换到项目目录  
cd D:\other_people_l\DX\competing_method\improve_3\Incomplete\3.MARLFS\dGLCN-main\dGLCN-main\codes\MARLFS
  
# 激活虚拟环境  
D:\other_people_l\DX\competing_method\improve_3\Incomplete\3.MARLFS\venv\Scripts\activate.ps1

# 运行Python脚本  
python .\a2c+mask_embedding.py --index=$Index