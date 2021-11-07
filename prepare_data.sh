workspace='rockyo'
cd /home/$workspace/Chemei-PR

echo 'Converting dcm raw dataset into jpg files...'

python3 src/preprocess/convert_jpg.py

echo 'Preparing VOC dataset...'

rm -rf ./data/PSPF_voc_data

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/T1c/ ./data/PSPF_voc_data/PR/T1c --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/T1c/ ./data/PSPF_voc_data/non_PR/T1c --labels ./data/segment_label.txt --noviz


echo 'Start segmentation...'

python3 src/preprocess/segment_tumor.py