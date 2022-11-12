
# ######################## 环境配置 ##################################

# 将fcntl.py文件放到环境的lib文件下
# 安装包：
                链接：https://www.lfd.uci.edu/~gohlke/pythonlibs/
                curses 
# 在blessings中注释掉 #from termios import TIOCGWINSZ



# ####################### 数据准备 ####################################

#   1. 实际的标注数据：
#      全部数据：http://www.cvlibs.net/download.php?file=data_depth_annotated.zip
#      小数据集：
#   2. 实际训练的图像数据：
#       datasets/kitti_archives_to_download.txt   中下载一个即可

#   3. 更改训练和验证TXT文件  --- 加载数据的位置

#   4. 需要下载预训练模型

# 数据包括：depth，image


# #####################  运行文件 #######################################

# 训练  KITTI 
python train.py --distributed --batch_size 16 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0,1,2,3

# 评估 KITTI
python eval.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --evaluate --batch_size 1 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0

# 演示单张图片的预测)
python demo.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --img_dir ./your/file/path/filename --pretrained KITTI --cuda --gpu_num 0

















###### 报错 #############################

1.  from prompt_toolkit.formatted_text import PygmentsTokens
ModuleNotFoundError: No module named 'prompt_toolkit.formatted_text'
合适的版本

3.数据的问题：自己间的数据集中有 换行符

2.分布式训练的问题
https://blog.csdn.net/qq_50027359/article/details/126779341?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-126779341-blog-121532525.pc_relevant_3mothn_strategy_recovery&spm=1001.2101.3001.4242.3&utm_relevant_index=7

train_sample 出现问题



#### 
option.py 中 是args，即一些参数的设计
datasets_list.py ： 
                 mydataset类：数据加载类
                 Transform类：进行数据归一化加tensor

model.py:
              定义模型的文件
              LRDN：网络模型
                     特征提取模块：ResNet101，MobileNetV2，。。。
                     边界提取模块：简单上下采样相减操作
                     编码模块： Lap_decoder_lv5，Lap_decoder_lv6 （不同的下采样次数）
                                Dilated_bottleNeck :  ASPP-空洞卷积，增加感受野  
                                WSconv
                                myconv
utils.py: 
           损失函数的定义 depth map predicted from a single image using a Multi-scale deep network
            
                    
                     

                     


