'''
Author: wtyo1768 wtyo1768@gmail.com
Date: 2021-10-08 14:10:09
LastEditors: wtyo1768 wtyo1768@gmail.com
LastEditTime: 2022-08-01 15:25:52
FilePath: /Chemei-PR/cfg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Comet settings
COMET_APT_KEY = 'PxM1YJCAf35smLk895Fs9wobK'
COMET_PROJECT_NAME = "pr-classifier"
COMET_WORK_SPACE = "wtyo1768" 

# Data directory 
xls_file = '/home/rockyo/Chemei-PR/data/PSPF meningioma 20211005.xls'
dcm_folder_path = "/home/rockyo/Chemei-PR/data/PSPF20210904"
jpg_folder_path = "/home/rockyo/Chemei-PR/data/PSPF20210904"
voc_data_path = "/home/rockyo/Chemei-PR/data/PSPF_voc_data"

segmented_img_dir = "/home/rockyo/Chemei-PR/data/segmented_img"

# Hyperparams
NO_SEGMENTED = False
NUM_RANDOM_STATE = 15
K_FOLD = 5
IMAGE_SIZE = 96
SEEDS = [74,53,24,12,37,39,46,9,32,99,84,56,96,47,23]	