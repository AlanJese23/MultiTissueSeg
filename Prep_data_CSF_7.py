import numpy as np   
import nibabel as nib
import glob, os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import argparse
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_rows', None)

# import pandas as pd


def main(input_data_path, output_data_path, mask_file, GA_info = None, input_size = 192, split_method = 'train_valid_split', n_fold = 5, fold_path = '0.txt'):

    # Path of Dataset
    # You need to modify code to properly search MRI data and Segmentation label
    MR_list = np.asarray(sorted(glob.glob(input_data_path + '/Upsample_MR' + '/*.nii')))
    GT_list = np.asarray(sorted(glob.glob(input_data_path + '/Upsample_GT' + '/*.nii')))

    #print(MR_list)
    #print(GT_list)


    mask_im = np.squeeze(nib.load(mask_file).get_fdata())
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
    repre_size[1] = input_size

   
    
    if split_method == 'train_valid_split':
        # Train-validation split
        output_dir = output_data_path + '/split'

        os.makedirs(output_dir+"/axi/train", exist_ok=True)
        os.makedirs(output_dir+"/axi/valid", exist_ok=True)
        os.makedirs(output_dir+"/cor/train", exist_ok=True)
        os.makedirs(output_dir+"/cor/valid", exist_ok=True)
        os.makedirs(output_dir+"/sag/train", exist_ok=True)
        os.makedirs(output_dir+"/sag/valid", exist_ok=True)

        #stratification
        MR_tbl = pd.DataFrame({"ID_file": [os.path.basename(file).replace("recon_to31_nuc_","").replace(".nii","") for file in MR_list],
                       "MR": MR_list})
        #print(MR_tbl)
        GT_tbl = pd.DataFrame({"ID_file": [os.path.basename(file).replace("new_seg_7_","").replace(".nii","") for file in GT_list],
                        "GT": GT_list})
        #print(GT_tbl)
        # Load information from CSV
        info_tbl = pd.read_csv("data_info_final_all.csv")
        info_tbl2 = pd.DataFrame({"ID_file": [case.replace("//","").replace("/",'_')  for case in info_tbl["ID"]],
                                "GA": [np.round(ga) for ga in info_tbl[" GA"]], "ID": info_tbl["ID"]})
        print(info_tbl2)
        # Process information table

        #info_tbl["ID_file"] = info_tbl["ID"]
        #info_tbl["GA"] = info_tbl[" GA"].round()


        # Merge data
        data_tbl = pd.merge(MR_tbl, GT_tbl, on="ID_file", validate="one_to_one")

        
        #data_tbl = pd.merge(data_tbl, info_tbl2[["ID_file", "GA", "ID"]], on="ID_file", validate="one_to_one")
        
        #data_tbl.to_csv("Temp.csv")

        total_tbl = pd.merge(data_tbl, info_tbl2, on="ID_file")
        print(total_tbl)
        total_tbl.to_csv("Temp.csv")

        MR_list_total = total_tbl["MR"]
        GT_list_total = total_tbl["GT"]
        GA_list_total = total_tbl["GA"]
        
        #MR_list_train, MR_list_valid, GT_list_train, GT_list_valid  = list(train_test_split(MR_list_total, GT_list_total, test_size=0.4, random_state=2, stratify=GA_list_total))
        MR_list_train, MR_list_valid, GT_list_train, GT_list_valid  = list(train_test_split(MR_list_total, GT_list_total, test_size=0.1, random_state=1))
        
        with open('Train_list.txt', 'w') as f:
            for line in MR_list_train:
                f.write(f"{line}\n")            
        with open('Validation_list.txt', 'w') as f:
            for line in MR_list_valid:
                f.write(f"{line}\n")
        process_data(MR_list_train, GT_list_train, MR_list_valid, GT_list_valid, output_dir, input_size, mask_im, mask_min, mask_max, repre_size, flip=True)

    elif split_method == 'kf':
        # K-fold cross-validation
        output_dir = output_data_path + '/kfold' + str(n_fold)
        
        # output_dir = 'Highres_dataset/kf'
        
        for fold_idx in range(0,n_fold):
            os.makedirs("{}/{}/axi/train".format(output_dir, fold_idx), exist_ok=True)
            os.makedirs("{}/{}/axi/valid".format(output_dir, fold_idx), exist_ok=True)
            os.makedirs("{}/{}/cor/train".format(output_dir, fold_idx), exist_ok=True)
            os.makedirs("{}/{}/cor/valid".format(output_dir, fold_idx), exist_ok=True)
            os.makedirs("{}/{}/sag/train".format(output_dir, fold_idx), exist_ok=True)
            os.makedirs("{}/{}/sag/valid".format(output_dir, fold_idx), exist_ok=True)
        
        kf = KFold(n_splits=n_fold, random_state=1, shuffle=True)

        # fold_info = list(kf.split(MR_list))
        ## The code to process_data for each fold need to be coded here.
        # MR_list_train, MR_list_valid, GT_list_train, GT_list_valid  = list(train_test_split(MR_list, GT_list, test_size=0.1, random_state=1))
        # process_data(MR_list_train, GT_list_train, MR_list_valid, GT_list_valid, output_dir, input_size, mask_im, mask_min, mask_max, repre_size, flip=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(MR_list)):
            print(fold_idx, train_idx, valid_idx)
            print(fold_idx, len(train_idx)/(len(train_idx) + len(valid_idx)), len(valid_idx)/(len(train_idx) + len(valid_idx)))
            process_fold(fold_idx, train_idx, valid_idx, MR_list, GT_list, output_dir, input_size, mask_im, mask_min, mask_max, repre_size)
        
    else:
        # Stratified K-fold (group with GA)
        fold_group = np.loadtxt(GA_info).astype(int)
        skf = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
        fold_info = list(skf.split(MR_list, fold_group))
        ## The code to process_data for each fold need to be coded here.


def process_fold(fold_idx, train_idx, valid_idx, MR_list_total, GT_list_total, output_dir, input_size, mask_im, mask_min, mask_max, repre_size):
    print("Processing {} fold".format(fold_idx))
    MR_list_train_fold = MR_list_total[train_idx]
    GT_list_train_fold = GT_list_total[train_idx]

    MR_list_valid_fold = MR_list_total[valid_idx]
    GT_list_valid_fold = GT_list_total[valid_idx]
    
    # Assuming process_data is a function you've defined elsewhere
    process_data(MR_list_train_fold, GT_list_train_fold, MR_list_valid_fold, GT_list_valid_fold, f"{output_dir}/{fold_idx}", input_size, mask_im, mask_min, mask_max, repre_size, flip=True)
    
def get_MR_data(img, label, mask_img, mask_min, mask_max):
    
    img = np.squeeze(nib.load(img).get_fdata()) * mask_img
    img = img[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
    # clip pixel less than 2% or over 98%
    loc = np.where(img<np.percentile(img,2))
    img[loc]=0
    loc = np.where(img>np.percentile(img,98))
    img[loc]=0
    loc = np.where(img)
    img[loc] = (img[loc] - np.mean(img[loc])) / np.std(img[loc])
    label = np.squeeze(nib.load(label).get_fdata())
    label = label[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
    return img, label

def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],
                                 array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis],
                                 array[:,:,:,7,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],
                                 array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis],
                                 array[:,:,:,7,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array

def preprocess_volume(MR_volume, GT_volume, input_size, view, mask_img, mask_min, mask_max, repre_size, flip):

    # For each input with designated view
    img, label = get_MR_data(MR_volume, GT_volume, mask_img, mask_min, mask_max)

    # Initialization
    if view == 'axi':
        X_image = np.zeros([repre_size[2], input_size, input_size,1])
        Y_label = np.zeros([repre_size[2], input_size, input_size,8])
    elif view == 'cor':
        X_image = np.zeros([repre_size[1], input_size, input_size,1])
        Y_label = np.zeros([repre_size[1], input_size, input_size,8])
    elif view == 'sag':
        X_image = np.zeros([repre_size[0], input_size, input_size,1])
        Y_label = np.zeros([repre_size[0], input_size, input_size,5])
    
    if view == 'axi':
        img2 = np.pad(img,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                        (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                        (0,0)), 'constant')
        X_image[:repre_size[2],:,:,0]= np.swapaxes(img2,2,0)
        img2 = np.pad(label,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                            (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                            (0,0)), 'constant')
        img2 = np.swapaxes(img2,2,0)
    elif view == 'cor':
        img2 = np.pad(img,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                        (0,0),
                        (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))),'constant')
        X_image[:repre_size[1],:,:,0]= np.swapaxes(img2,1,0)
        img2 = np.pad(label,((int(np.floor((input_size-img.shape[0])/2)),int(np.ceil((input_size-img.shape[0])/2))),
                            (0,0),
                            (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))),'constant')
        img2 = np.swapaxes(img2,1,0)
    elif view == 'sag':
        img2 = np.pad(img,((0,0),
                        (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                        (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2)))), 'constant')
        X_image[:repre_size[0],:,:,0]= img2
        img2 = np.pad(label,((0,0),
                            (int(np.floor((input_size-img.shape[1])/2)),int(np.ceil((input_size-img.shape[1])/2))),
                            (int(np.floor((input_size-img.shape[2])/2)),int(np.ceil((input_size-img.shape[2])/2))),), 'constant')

    if (view == 'axi') | (view == 'cor'):
        img3 = np.zeros_like(img2)
        back_loc = np.where(img2<0.5)
        # left_in_loc = np.where((img2>160.5)&(img2<161.5))
        # right_in_loc = np.where((img2>159.5)&(img2<160.5))
        left_iz_loc = np.where((img2>160.5)&(img2<161.5))
        right_iz_loc = np.where((img2>159.5)&(img2<160.5))
        left_plate_loc = np.where((img2>0.5)&(img2<1.5))
        right_plate_loc = np.where((img2>41.5)&(img2<42.5))
        vent_left = np.where((img2>131.5)&(img2<132.5))
        vent_right = np.where((img2>132.5)&(img2<133.5))
        csf = np.where((img2>17.5)&(img2<18.5))

        img3[back_loc]=1

    elif view == 'sag':
        img3 = np.zeros_like(img2)
        back_loc = np.where(img<0.5)
        # in_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
        iz_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
        plate_loc = np.where((img2>0.5)&(img2<1.5)|(img2>41.5)&(img2<42.5))
        vent_loc = np.where((img2>131.5)&(img2<132.5)|(img2>132.5)&(img2<133.5))
        csf_loc = np.where((img2>17.5)&(img2<18.5))
        img3[back_loc]=1

    if view == 'axi':
        Y_label[:repre_size[2],:,:,0]=img3
        img3[:]=0

        img3[left_iz_loc]=1
        Y_label[:repre_size[2],:,:,1]=img3
        img3[:]=0

        img3[right_iz_loc]=1
        Y_label[:repre_size[2],:,:,2]=img3
        img3[:]=0

        img3[left_plate_loc]=1
        Y_label[:repre_size[2],:,:,3]=img3
        img3[:]=0

        img3[right_plate_loc]=1
        Y_label[:repre_size[2],:,:,4]=img3
        img3[:]=0

        img3[vent_left]=1
        Y_label[:repre_size[2],:,:,5]=img3
        img3[:]=0

        img3[vent_right]=1
        Y_label[:repre_size[2],:,:,6]=img3
        img3[:]=0

        img3[csf]=1
        Y_label[:repre_size[2],:,:,7]=img3
        img3[:]=0

    elif view == 'cor':
        Y_label[:repre_size[1],:,:,0]=img3
        img3[:]=0

        img3[left_iz_loc]=1
        Y_label[:repre_size[1],:,:,1]=img3
        img3[:]=0
        img3[right_iz_loc]=1
        Y_label[:repre_size[1],:,:,2]=img3
        img3[:]=0

        img3[left_plate_loc]=1
        Y_label[:repre_size[1],:,:,3]=img3
        img3[:]=0
        
        img3[right_plate_loc]=1
        Y_label[:repre_size[1],:,:,4]=img3
        img3[:]=0

        img3[vent_left]=1
        Y_label[:repre_size[1],:,:,5]=img3
        img3[:]=0

        img3[vent_right]=1
        Y_label[:repre_size[1],:,:,6]=img3
        img3[:]=0

        img3[csf]=1
        Y_label[:repre_size[1],:,:,7]=img3
        img3[:]=0

    elif view == 'sag':
        Y_label[:repre_size[0],:,:,0]=img3
        img3[:]=0

        img3[iz_loc]=1
        Y_label[:repre_size[0],:,:,1]=img3
        img3[:]=0

        img3[plate_loc]=1
        Y_label[:repre_size[0],:,:,2]=img3
        img3[:]=0

        img3[vent_loc]=1
        Y_label[:repre_size[0],:,:,3]=img3
        img3[:]=0

        img3[csf_loc]=1
        Y_label[:repre_size[0],:,:,4]=img3
        img3[:]=0

    if flip:
        if view == 'axi':
            X_image=np.concatenate((X_image,X_image[:,::-1,:,:]),axis=0)
            Y_label=np.concatenate((Y_label,Y_label[:,::-1,:,:]),axis=0)
            X_image=np.concatenate((X_image, axfliper(X_image)),axis=0)
            Y_label=np.concatenate((Y_label, axfliper(Y_label, 1)),axis=0)
        elif view == 'cor':
            X_image=np.concatenate((X_image,X_image[:,:,::-1,:]),axis=0)
            Y_label=np.concatenate((Y_label,Y_label[:,:,::-1,:]),axis=0)
            X_image=np.concatenate((X_image, cofliper(X_image)),axis=0)
            Y_label=np.concatenate((Y_label, cofliper(Y_label, 1)),axis=0)
        elif view == 'sag':
            X_image = np.concatenate((X_image, X_image[:,:,::-1,:], X_image[:,::-1,:,:]),axis=0)
            Y_label = np.concatenate((Y_label, Y_label[:,:,::-1,:], Y_label[:,::-1,:,:]),axis=0)
        
    return X_image, Y_label
  
def process_data(MR_list_train, GT_list_train, MR_list_valid, GT_list_valid, output_dir, input_size, mask_im, mask_min, mask_max, repre_size, flip):

    # Training data
    print("Processing training set")
    for (MR_volume, GT_label) in tqdm(zip(MR_list_train, GT_list_train), total=len(MR_list_train)):
        volume_name = os.path.splitext(os.path.basename(MR_volume))[0]
    
        # Axial plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='axi', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        #print("X_MR shape: ", X_MR.shape)
        #print("Y_GT shape: ", Y_GT.shape)
   
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/axi/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Coronal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='cor', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/cor/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Sagittal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='sag', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/sag/train/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

    # Validation data
    print("Processing validation set")
    for (MR_volume, GT_label) in tqdm(zip(MR_list_valid, GT_list_valid), total=len(MR_list_valid)):
        volume_name = os.path.splitext(os.path.basename(MR_volume))[0]
    
        # Axial plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='axi', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/axi/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Coronal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='cor', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/cor/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])

        # Sagittal plane preprocessing
        X_MR, Y_GT = preprocess_volume(MR_volume, GT_label, input_size=input_size, view='sag', mask_img=mask_im, mask_min=mask_min, mask_max=mask_max, repre_size=repre_size, flip=flip)
        
        for slice_idx in range(0, X_MR.shape[0]):
            
            save_name = output_dir + "/sag/valid/" + volume_name + "_{:03}".format(slice_idx)
            
            np.save(save_name + "_MR", X_MR[slice_idx])
            np.save(save_name + "_GT", Y_GT[slice_idx])
            
if __name__ == "__main__":
    # Data preparation code for CP segmentation of high resolution recon.
    # Only code train-test-split (train:90%, validation:10%) is implemented yet.

    parser = argparse.ArgumentParser('   ==========   Data preparation code for fetal Attention U-Net for CP segmentation on high resolution recontruction script made by Sungmin You (11.10.23 ver.0)   ==========   ')
    parser.add_argument('-Input_dir', '--i',action='store',dest='input_dir',type=str, required=True, help='Input path for raw MRI files(*.nii)')
    parser.add_argument('-Output_dir', '--o',action='store',dest='output_dir',type=str, required=True, help='Output path for organized dataset(*.npy)')
    parser.add_argument('-mask', '--mask', action='store',dest='mask',type=str, default='down_mask-31_dil5.nii', help='mask file to make dictionary and intensity normalize')
    parser.add_argument('-split', '--split', dest='split_method', type=str, default='train_valid_split', help='training/validation data splitting method')
    parser.add_argument('-fold', '--fold', dest='fold_path', type=str, help='fold text path')
    #parser.add_argument('-f', '--num_fold',action='store',dest='num_fold',default=10, type=int, help='number of fold for training')
    #parser.add_argument('-fi', '--stratified_info_file',action='store', dest='stratified_info',type=str, help='information for stratified fold')
    parser.add_argument('-is', '--input_shape',action='store', dest='input_shape',type=int, default=192, help='Input size')
    args = parser.parse_args()

    main(input_data_path = args.input_dir, output_data_path = args.output_dir, mask_file = args.mask, input_size = args.input_shape,
         split_method = args.split_method, fold_path = args.fold_path)
    # python Prep_high_res_data.py --Input_data_dir Upsample_0.5 --Output_data_dir Upsample_dataset2 --mask down_mask-31_dil5.nii --input_shape 192
    # python Prep_high_res_data.py --Input_data_dir . --Output_data_dir Subject_n64_LAS --mask down_mask-31_dil5.nii --input_shape 192

    # python3 prep_data.py --input data/set1/set1_nii --output data/set1/fold0_npy --mask down_mask-31_dil5.nii --input_shape 192 --split manual_strat --fold data/set1/fold_0.txt
    # python3 prep_data.py --input data/set40/set40_nii --output data/set40/auto2_npy --mask down_mask-31_dil5.nii --input_shape 192 --split train_valid_split
