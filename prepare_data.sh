workspace='rockyo'

cd /home/$workspace/Chemei-PR

echo 'Converting dcm raw dataset into jpg files...'

python3 preprocess/convert_jpg.py

echo 'Preparing VOC dataset...'

rm -rf ./data/PSPF_voc_data

# Using labelme package to create VOC segmentation dataset for each MRI type
# T1c
../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/T1/ ./data/PSPF_voc_data/PR/T1 --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/T1/ ./data/PSPF_voc_data/non_PR/T1 --labels ./data/segment_label.txt --noviz

# T1
../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/T1/ ./data/PSPF_voc_data/PR/T1 --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/T1/ ./data/PSPF_voc_data/non_PR/T1 --labels ./data/segment_label.txt --noviz

# T2
../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/T2/ ./data/PSPF_voc_data/PR/T2 --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/T2/ ./data/PSPF_voc_data/non_PR/T2 --labels ./data/segment_label.txt --noviz

# Flair
../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/Flair/ ./data/PSPF_voc_data/PR/Flair --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/Flair/ ./data/PSPF_voc_data/non_PR/Flair --labels ./data/segment_label.txt --noviz

# copy source MRI segment label to taget 
python3 preprocess/copy_segment_label.py

echo 'Start segmentation...'

# segment tumor region
python3 preprocess/segment_tumor.py