workspace='rockyo'


cd /home/$workspace/Chemei-PR


# python3 convert_jpg.py

# Prepare VOC dataset

rm -rf ./data/PSPF_voc_data

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/PR_jpg/T1c/ ./data/PSPF_voc_data/PR/T1c --labels ./data/segment_label.txt --noviz

../labelme/examples/semantic_segmentation/labelme2voc.py \
./data/PSPF20210904/non_PR_jpg/T1c/ ./data/PSPF_voc_data/non_PR/T1c --labels ./data/segment_label.txt --noviz


# Segmentation

python3 segment_tumor.py