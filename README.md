
1. 创建环境

   ```
   conda create -n medical python=3.7.11 //创建一个名为medical的python版本为3.7.11的conda环境
   
   source activate medical //登入环境
   ```

2. 安装python包

   首先安装pytorch，其次是其它包

   ```
   pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
   pip install -r requirements.txt
   ```

3. 运行
   ```
   cd src
   ./run_cmeee.sh
   ```
   针对三个不同实验，修改run_cmeee.sh中的task_id即可


