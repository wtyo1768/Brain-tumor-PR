from typing import Counter
import pandas as pd
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import os
import cv2


from cfg import *
# from loader import read_xls

excel_data = pd.read_excel(xls_file, sheet_name="Sheet2")
# excel_data = read_xls()
'''
#ID : [T1 + C , T1 , T2 + C , FLAIR]
dict = {ID : [(srs , img) , (srs , img) , (srs , img) , (srs , img)] }
'''


def appendd(iid, img_dict, col_name):
    srs, img = (excel_data.loc[index, col_name]).split("/")
    temp = int(srs), int(img)
    img_dict[iid].append(temp)


def convert(ds, store_path, patient_id):
    pixel_array_numpy = ds.pixel_array.astype(float)
    rescaled_img = (np.maximum(pixel_array_numpy, 0) /
                    pixel_array_numpy.max()) * 255
    # image = image.replace('.dcm', '.jpg')
    rescaled_img = cv2.cvtColor(
        rescaled_img.astype(np.float32), cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(store_path, f'{patient_id}.jpg'), rescaled_img)


img_dict = {}
#non_dict = {}
#pr_dict = {}
for index in range(0, excel_data.shape[0]):
    # naming
    iid = excel_data.loc[index, "病歷號 (yellow in WHO grade I file) "]
    pr_status = excel_data.loc[index, "Progression/Recurrence (Yes:1 No:0)"]
    img_dict[iid] = []
    appendd(iid, img_dict, "T1+C (srs/img)")
    appendd(iid, img_dict, "T1 (srs/img)")
    appendd(iid, img_dict, "T2 (srs/img)")
    appendd(iid, img_dict, "FLAIR (srs/img)")


print(img_dict)
print("******************************************************")


folder_path = f"{dcm_folder_path}/non PR"
jpg_folder_path = f"{dcm_folder_path}/non_PR_jpg"

jpg_folder_path2 = f"{dcm_folder_path}/PR_jpg"
dcm_folder_path2 = f"{dcm_folder_path}/PR"

error_patient = []
MRI_TYPE = ['T1', 'T1c', 'Flair', 'T2']
for d in MRI_TYPE:
    os.makedirs(f'{jpg_folder_path}/{d}')  # for non PR
    os.makedirs(f'{jpg_folder_path2}/{d}')  # for PR

print("convert start : wait......")
# ******************************convert block***********************************************
dir_count = 0
path_list = os.listdir(folder_path)
while(True):
    for patient_id in path_list:
        if patient_id in ['16962871', '14185785']:  # data loss
            continue
        print(type(os.listdir(folder_path)))
        print(patient_id)
        picture_count = set()
        T1c = img_dict[int(patient_id)][0]
        T1 = img_dict[int(patient_id)][1]
        T2 = img_dict[int(patient_id)][2]
        Flair = img_dict[int(patient_id)][3]

        if dir_count == 0:  # non_PR
            patient_path = os.path.join(folder_path, patient_id)
            for n, image in enumerate(os.listdir(patient_path)):
                ds = pydicom.dcmread(os.path.join(patient_path, image))
                # non _dict (every image should be judged 4 times)

                if ds.SeriesNumber == T1[0] and ds.InstanceNumber == T1[1]:
                    convert(ds, f'{jpg_folder_path}/T1', patient_id)
                    picture_count.add('T1')
                if ds.SeriesNumber == T1c[0] and ds.InstanceNumber == T1c[1]:
                    convert(ds, f'{jpg_folder_path}/T1c', patient_id)
                    picture_count.add('T1c')
                if ds.SeriesNumber == T2[0] and ds.InstanceNumber == T2[1]:
                    convert(ds, f'{jpg_folder_path}/T2', patient_id)
                    picture_count.add('T2')
                if ds.SeriesNumber == Flair[0] and ds.InstanceNumber == Flair[1]:
                    convert(ds, f'{jpg_folder_path}/Flair', patient_id)
                    picture_count.add('Flair')

        elif dir_count == 1:  # PR
            patient_path = os.path.join(dcm_folder_path2, patient_id)
            for n, image in enumerate(os.listdir(patient_path)):
                ds = pydicom.dcmread(os.path.join(
                    patient_path, image), force=True)
                try:
                    if ds.SeriesNumber == T1[0] and ds.InstanceNumber == T1[1]:
                        convert(ds, f'{jpg_folder_path2}/T1', patient_id)
                        picture_count.add('T1')
                    if ds.SeriesNumber == T1c[0] and ds.InstanceNumber == T1c[1]:
                        convert(ds, f'{jpg_folder_path2}/T1c', patient_id)
                        picture_count.add('T1c')
                    if ds.SeriesNumber == T2[0] and ds.InstanceNumber == T2[1]:
                        convert(ds, f'{jpg_folder_path2}/T2', patient_id)
                        picture_count.add('T2')
                    if ds.SeriesNumber == Flair[0] and ds.InstanceNumber == Flair[1]:
                        convert(ds, f'{jpg_folder_path2}/Flair', patient_id)
                        picture_count.add('Flair')
                except:
                    print(os.error)
                    error_patient.append(patient_id)

        if len(picture_count) != 4:
            error_patient.append(patient_id)

    dir_count += 1
    if dir_count >= 2:
        break
    path_list[:] = os.listdir(dcm_folder_path2)
# *******************************************************************************************************
if error_patient != [] and error_patient != ['10027124']:
    print("error patient", error_patient)
print("convert successfully!")
# another data error found -> patient 10027124 (dicomfile loss SeriesNumber data)
