from cfg import *
import shutil
import os


src_type = 'T1'
dst_type = ['T1c', 'T2', 'Flair']


# Copy segment label
for patient_class in ['PR', 'non_PR']: 
    for dtype in dst_type:
        src_dir = f'{voc_data_path}/{patient_class}/{src_type}/SegmentationClassPNG'
        out_dir = f'{voc_data_path}/{patient_class}/{dtype}/SegmentationClassPNG'
        assert os.path.isdir(src_dir)

        shutil.copytree(src_dir, out_dir)



dst_type = ['T1c', 'T2', 'Flair']
# Copy image 
for patient_class in ['PR', 'non_PR']: 
    for dtype in dst_type:        
        # Copy image
        src_dir = f'{jpg_folder_path}/{patient_class}_jpg/{dtype}'
        out_dir = f'{voc_data_path}/{patient_class}/{dtype}/JPEGImages'
        print(src_dir)
        assert os.path.isdir(src_dir)
        shutil.copytree(src_dir, out_dir)