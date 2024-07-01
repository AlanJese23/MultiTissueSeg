import os, glob
import tensorflow as tf
from CSF_Vent_utils_7 import *
from datetime import datetime
import argparse

def main(args):
    
    # Use single GPU for the training
    if args.gpu != 'multi':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    n_batch=args.bsize
    input_dim=args.isize
    n_epoch = args.epoch
    learning_rate = args.lr

    if args.view == 'cor':
        output_channel = 8
    elif args.view == 'axi':
        output_channel = 8
    elif args.view == 'sag':
        output_channel = 5


    print('\n\n')
    print('\t\t Dataset folder: \t\t\t\t'+os.path.realpath(args.data_dir + args.split + args.view))
    print('\t\t Model weights save location: \t\t\t'+args.weight_loc)
    print('\t\t Training history save location: \t\t\t'+args.hist_loc)
    print('\t\t input_shape: \t\t\t\t\t\t'+str(args.isize))
    print('\t\t batch_size: \t\t\t\t\t\t'+str(args.bsize))
    print('\t\t Output channel size: \t\t\t\t\t'+str(output_channel))
    print('\t\t Learning rate: \t\t\t\t\t'+str(args.lr))
    print('\t\t Training loss: \t\t\t\t\t'+args.loss)
    print('\t\t Epochs: \t\t\t\t\t\t'+str(args.epoch))
    print('\t\t GPU number: \t\t\t\t\t\t'+args.gpu)
    print('\t\t View: \t\t\t\t\t\t'+args.view)

    # Data load
    MR_train_list = sorted(glob.glob(args.data_dir + '/' + args.split + '/' +  args.view + "/train/*_MR.npy"))
    GT_train_list = sorted(glob.glob(args.data_dir + '/' + args.split + '/' + args.view + "/train/*_GT.npy"))
    print("Training data number: " + str(len(MR_train_list)))

    MR_valid_list = sorted(glob.glob(args.data_dir + '/' + args.split + '/' + args.view + "/valid/*_MR.npy"))
    GT_valid_list = sorted(glob.glob(args.data_dir + '/' + args.split + '/' + args.view + "/valid/*_GT.npy"))
    print("Validation data number: " + str(len(MR_valid_list)))

    os.makedirs(args.hist_loc, exist_ok=True)
    os.makedirs(args.weight_loc, exist_ok=True)

    Training_loader = DataGenerator(MR_list=MR_train_list, GT_list=GT_train_list, out_channels=output_channel, input_dim=input_dim, batch_size=n_batch)
    Valid_loader = DataGenerator(MR_list=MR_valid_list, GT_list=GT_valid_list, out_channels=output_channel, input_dim=input_dim, batch_size=n_batch)

    callbacks=make_callbacks(args.weight_loc+'/'+ args.view+'.h5', args.hist_loc+'/'+ args.view+'.tsv')

    if args.gpu == 'multi':
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = Unet_network([input_dim,input_dim,1], output_channel, loss=args.loss, metrics=args.metric, style='basic', ite=3, depth=4, dim=32, init='he_normal', acti='elu', lr=learning_rate, dropout=0).build()
    else:
        model = Unet_network([input_dim,input_dim,1], output_channel, loss=args.loss, metrics=args.metric, style='basic', ite=3, depth=4, dim=32, init='he_normal', acti='elu', lr=learning_rate, dropout=0).build()
        
    start_time = datetime.now()
    model.fit(Training_loader, batch_size = n_batch, epochs = n_epoch, validation_data=Valid_loader, verbose=2, workers=16, use_multiprocessing=True, callbacks=callbacks)
    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))


if __name__== '__main__':
        
    parser = argparse.ArgumentParser('   ==========   Fetal Attention U-Net for CP segmentation on high resolution recontruction script made by Sungmin You (11.10.23 ver.0)   ==========   ')
    parser.add_argument('-Data_dir', '--dir',action='store',dest='data_dir',type=str, required=True, help='Path for training set')
    parser.add_argument('-wl', '--weight_save_location', action='store',dest='weight_loc', type=str, default='weights/', help='Output folder name, default: weights/')
    parser.add_argument('-hl', '--history_save_location', action='store',dest='hist_loc', type=str, default='history/', help='Output folder name, default: history/')
    parser.add_argument('-is', '--input_shape',action='store', dest='isize',type=int, default = 192, help='Input size 192')
    parser.add_argument('-bs', '--batch_size',action='store', dest='bsize',type=int, default=32, help='batch size')
    parser.add_argument('-e', '--epoch',action='store',dest='epoch',default=100,  type=int, help='Number of epoch for training')
    parser.add_argument('-lr', '--learning_rate', default=1e-5, action='store',dest='lr', type=float, help='Learning rate')
    parser.add_argument('-l', '--loss', choices=['hyb_loss', 'hyb_loss2', 'ori_dice_loss', 'dice_loss','dis_dice_loss'], default='hyb_loss', action='store',dest='loss', type=str, help='Loss function')
    parser.add_argument('-m', '--metric', choices=['dice_coef', 'dis_dice_coef'], default=['dice_coef'], nargs='*', action='store',dest='metric', help='Eval metric')
    parser.add_argument('-gpu', '--gpu_number',action='store',dest='gpu',type=str, default='0',help='Select GPU')
    parser.add_argument('-split', '--data_split',action='store',dest='split',type=str, default='split', help='Method used to split dataset')
    parser.add_argument('-view', choices=['axi', 'cor', 'sag'], action='store', dest='view', required=True, help='View for training')
    args = parser.parse_args()    

    main(args)
    
    #python Training_high_res_fetal_cp_seg.py.py --Dataset_dir Subject_n29 --epoch 1 --gpu_number 0 -view axi
