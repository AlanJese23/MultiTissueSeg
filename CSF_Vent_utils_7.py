import tensorflow as tf
import numpy as np   
import nibabel as nib

class Unet_network:
    def __init__(self, input_shape, out_ch, loss='dice_loss', metrics=['dice_coef', 'dis_dice_coef'], style='basic', ite=2, depth=4, dim=32,
                  weights='', init='he_normal',acti='elu',lr=1e-4, dropout=0):
        from keras.layers import Input
        self.style=style
        self.input_shape=input_shape
        self.out_ch=out_ch
        self.loss = loss
        self.metrics = metrics
        self.ite=ite
        self.depth=depth
        self.dim=dim
        self.init=init
        self.acti=acti
        self.weight=weights
        self.dropout=dropout
        self.lr=lr
        self.I = Input(input_shape)
        self.ratio = None
        self.kernel2 = None


    def conv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (3,3), padding='same', kernel_initializer=self.init)(x)
        return x

    def conv1_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(x)
        return x

    def tconv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2DTranspose
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2DTranspose(dim, 2, strides=2, padding='same', kernel_initializer=self.init)(x)
        return x

    def basic_block(self, inp, dim):
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        return inp

    def res_block(self, inp, dim):
        from keras.layers import Add
        inp2 = inp
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        cb2 = self.conv1_block(inp2,dim)
        return Add()([inp, cb2])

    def dense_block(self, inp, dim):
        from keras.layers import concatenate
        for i in range(self.ite):
            cb = self.conv_block(inp, dim)
            inp = concatenate([inp,cb])
        inp = self.conv1_block(inp,dim)
        return inp

    def RCL_block(self, inp, dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Add
        RCL=Conv2D(dim, (3,3), padding='same',kernel_initializer=self.init)
        conv=bn()(inp)
        conv=Activation(self.acti)(conv)
        conv=Conv2D(dim,(3,3),padding='same',kernel_initializer=self.init)(conv)
        conv2=bn()(conv)
        conv2=Activation(self.acti)(conv2)
        conv2=RCL(conv2)
        conv2=Add()([conv,conv2])
        for i in range(0, self.ite-2):
            conv2=bn()(conv2)
            conv2=Activation(self.acti)(conv2)
            conv2=Conv2D(dim, (3,3), padding='same',weights=RCL.get_weights())(conv2)
            conv2=Add()([conv,conv2])
        return conv2

    def att_gate(self, X, g, dim):
        from keras.layers import Activation, Conv2D, multiply, add
        xl = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(X)
        g = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(g)
        query = add([xl, g])
        f = Activation('relu')(query)
        psi_f = Conv2D(1, (1,1), padding='same', kernel_initializer=self.init)(f)
        coef_att = Activation('sigmoid')(psi_f)
        X_att = multiply([X, coef_att])
        return X_att

    def build_U(self, inp, dim, depth):
        from keras.layers import MaxPooling2D, concatenate, Dropout
        if depth > 0:
            if self.style == 'basic':
                x = self.basic_block(inp, dim)
            elif self.style == 'res':
                x = self.res_block(inp, dim)
            elif self.style == 'dense':
                x = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
            x2 = MaxPooling2D()(x)
            if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            x2 = self.build_U(x2, int(dim*2), depth-1)
            x2 = self.tconv_block(x2,int(dim*2))
            x = self.att_gate(x, x2, int(dim))
            x2 = concatenate([x,x2])
            if self.style == 'basic':
                x2 = self.basic_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'res':
                x2 = self.res_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'dense':
                x2 = self.dense_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'RCL':
                x2 = self.RCL_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        else:
            if self.style == 'basic':
                x2 = self.basic_block(inp, dim)
            elif self.style == 'res':
                x2 = self.res_block(inp, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        return x2

    def UNet(self):
        from keras.layers import Conv2D
        from keras.models import Model
        from tensorflow.keras.optimizers import Adam
        o = self.build_U(self.I, self.dim, self.depth)
        o = Conv2D(self.out_ch, 1, activation='softmax')(o)
        model = Model(inputs=self.I, outputs=o)

        
        if len(self.metrics)==2:
            model.compile(optimizer=Adam(learning_rate=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
        else:
            model.compile(optimizer=Adam(learning_rate=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])

        return model

    def build(self):
        return self.UNet()

    def dice_coef(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * K.round(y_pred), axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(K.round(y_pred), axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def dice_loss(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(K.pow(-K.log(dice[1:]),0.3))

    def ori_dice_loss(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return -K.mean(dice[1:])

    def dis_loss(self, y_true, y_pred):
        from keras import backend as K
        import numpy as np
        
        si=K.int_shape(y_pred)[-1]
        riter=2
        smooth = 0.001

        # cv2 circular kernel
        # import cv2
        # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))

        # np circular kernel @ Milton Candela 03/2024
        y, x = np.ogrid[-riter:riter+1, -riter:riter+1]
        kernel = np.zeros((2 * riter + 1, 2 * riter + 1), dtype=np.int8)
        kernel[x**2 + y**2 <= riter**2] = 1

        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        if self.kernel2 is None:
            self.kernel2=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,self.kernel2,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,self.kernel2,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        #y_true_s = y_true - y_true_s
        #y_pred_s = y_pred - y_pred_s
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.sum(K.pow(-K.log(dice[1:]),0.3))

    def dis_dice_coef(self, y_true, y_pred):
        from keras import backend as K
        import numpy as np
        si=K.int_shape(y_pred)[-1]
        riter=2
        smooth = 0.001

        # cv2 circular kernel
        # import cv2
        # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))

        # np circular kernel @ Milton Candela 03/2024
        y, x = np.ogrid[-riter:riter+1, -riter:riter+1]
        kernel = np.zeros((2 * riter + 1, 2 * riter + 1), dtype=np.int8)
        kernel[x**2 + y**2 <= riter**2] = 1

        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,kernel,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,kernel,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        #y_true_s = y_true - y_true_s
        #y_pred_s = y_pred - y_pred_s
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def hd_loss(self, y_true, y_pred):
        from keras import backend as K
        import numpy as np
        def in_func(in_tensor,in_kernel,in_f):
            return K.clip(K.depthwise_conv2d(in_tensor,in_kernel,data_format="channels_last",padding="same")-0.5,0,0.5)*in_f
        si=K.int_shape(y_pred)[-1]
        f_qp=K.square(y_true-y_pred)*y_pred
        f_pq=K.square(y_true-y_pred)*y_true
        p_b=K.cast(y_true,'float32')
        p_bc=1-p_b
        q_b=K.cast(y_pred>0.5,'float32')
        q_bc=1-q_b
        rtiter=0
        si=K.int_shape(y_pred)[-1]
        for riter in range(3,19,3):
            # cv2 circular kernel
            # import cv2
            # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))

            # np circular kernel @ Milton Candela 03/2024
            y, x = np.ogrid[-riter:riter+1, -riter:riter+1]
            kernel = np.zeros((2 * riter + 1, 2 * riter + 1), dtype=np.int8)
            kernel[x**2 + y**2 <= riter**2] = 1

            kernel=kernel/(np.sum(kernel))
            kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
            kernel=K.variable(kernel[:,:,:,np.newaxis])
            if rtiter == 0:
                loss=K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
            else:
                loss=loss+K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
        return K.mean(loss[1:])

    def hyb_loss(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.dis_loss(y_true, y_pred)
        return 0.1*h_loss + d_loss

    def hyb_loss2(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.hd_loss(y_true, y_pred)
        if self.ratio==None:
            loss = h_loss + d_loss
        else:
            loss = self.ratio * h_loss + (1-self.ratio) * d_loss
        self.ratio = d_loss / h_loss
        return loss

def reset_gpu():

    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    tf.compat.v1.reset_default_graph()

def set_gpu(gpu_num=0):
    import tensorflow as tf
    import os
    from keras import backend as K
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    K.set_session(tf.compat.v1.Session(config=config))

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

def make_dic(img_list, gold_list, mask, dim,flip=0):
    import numpy as np
    import nibabel as nib
    def get_data(img, label, mask_img, mask_min, mask_max):
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

    mask_im = np.squeeze(nib.load(mask).get_fdata())
    mask_min = np.min(np.where(mask_im), axis=1)
    mask_max = np.max(np.where(mask_im), axis=1)
    repre_size = mask_max-mask_min
    #print(repre_size)
    #max_repre = np.max(repre_size)
    max_repre = 192
    repre_size[1] = max_repre
    if dim == 'axi':
        dic = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,8])
    elif dim == 'cor':
        dic = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,8])
    elif dim == 'sag':
        dic = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,5])
    else:
        print('available: axi, cor, sag.   Your: '+dim)
        exit()

    for i in range(0, len(img_list)):
        img, label = get_data(img_list[i], gold_list[i], mask_im, mask_min, mask_max)
        if dim == 'axi':
            img2 = np.pad(img,((int(np.floor((max_repre-img.shape[0])/2)),int(np.ceil((max_repre-img.shape[0])/2))),
                               (int(np.floor((max_repre-img.shape[1])/2)),int(np.ceil((max_repre-img.shape[1])/2))),
                               (0,0)), 'constant')
            dic[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]= np.swapaxes(img2,2,0)
            img2 = np.pad(label,((int(np.floor((max_repre-img.shape[0])/2)),int(np.ceil((max_repre-img.shape[0])/2))),
                                 (int(np.floor((max_repre-img.shape[1])/2)),int(np.ceil((max_repre-img.shape[1])/2))),
                                 (0,0)), 'constant')
            img2 = np.swapaxes(img2,2,0)
        elif dim == 'cor':
            img2 = np.pad(img,((int(np.floor((max_repre-img.shape[0])/2)),int(np.ceil((max_repre-img.shape[0])/2))),
                               (0,0),
                               (int(np.floor((max_repre-img.shape[2])/2)),int(np.ceil((max_repre-img.shape[2])/2)))),'constant')
            dic[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]= np.swapaxes(img2,1,0)
            img2 = np.pad(label,((int(np.floor((max_repre-img.shape[0])/2)),int(np.ceil((max_repre-img.shape[0])/2))),
                                 (0,0),
                                 (int(np.floor((max_repre-img.shape[2])/2)),int(np.ceil((max_repre-img.shape[2])/2)))),'constant')
            img2 = np.swapaxes(img2,1,0)
        elif dim == 'sag':
            img2 = np.pad(img,((0,0),
                              (int(np.floor((max_repre-img.shape[1])/2)),int(np.ceil((max_repre-img.shape[1])/2))),
                              (int(np.floor((max_repre-img.shape[2])/2)),int(np.ceil((max_repre-img.shape[2])/2)))), 'constant')
            dic[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]= img2
            img2 = np.pad(label,((0,0),
                                (int(np.floor((max_repre-img.shape[1])/2)),int(np.ceil((max_repre-img.shape[1])/2))),
                                (int(np.floor((max_repre-img.shape[2])/2)),int(np.ceil((max_repre-img.shape[2])/2))),), 'constant')
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        if (dim == 'axi') | (dim == 'cor'):
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5))
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            vent_left = np.where((img2>131.5)&(img2<132.5))
            vent_right = np.where((img2>132.5)&(img2<133.5))
            csf = np.where((img2>17.5)&(img2<18.5))
            img3[back_loc]=1

        elif dim == 'sag':
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            in_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
            plate_loc = np.where((img2>0.5)&(img2<1.5)|(img2>41.5)&(img2<42.5))
            vent_loc = np.where((img2>131.5)&(img2<132.5)|(img2>132.5)&(img2<133.5))
            csf_loc = np.where((img2>17.5)&(img2<18.5))

            img3[back_loc]=1
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        if dim == 'axi':
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]=img3
            img3[:]=0

            img3[left_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,2]=img3
            img3[:]=0

            img3[left_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,3]=img3
            img3[:]=0

            img3[right_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,4]=img3
            img3[:]=0

            img3[vent_left]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,5]=img3
            img3[:]=0

            img3[vent_right]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,6]=img3
            img3[:]=0

            img3[csf]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,7]=img3
            img3[:]=0

        elif dim == 'cor':
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]=img3
            img3[:]=0

            img3[left_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,1]=img3
            img3[:]=0

            img3[right_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,2]=img3
            img3[:]=0
            
            img3[left_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,3]=img3
            img3[:]=0

            img3[right_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,4]=img3
            img3[:]=0

            img3[vent_left]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,5]=img3
            img3[:]=0

            img3[vent_right]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,6]=img3
            img3[:]=0

            img3[csf]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,7]=img3
            img3[:]=0

        elif dim == 'sag':
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[in_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[plate_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[vent_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[csf_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,4]=img3
            img3[:]=0
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    if flip:
        if dim == 'axi':
            dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
            seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
            dic=np.concatenate((dic, axfliper(dic)),axis=0)
            seg=np.concatenate((seg, axfliper(seg, 1)),axis=0)
        elif dim == 'cor':
            dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
            seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
            dic=np.concatenate((dic, cofliper(dic)),axis=0)
            seg=np.concatenate((seg, cofliper(seg, 1)),axis=0)
        elif dim == 'sag':
            dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
            seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    return dic, seg


def make_result(tmask, img_list, mask,result_loc,axis,ext=''):
    import nibabel as nib
    import numpy as np
    mask_im = nib.load(mask).get_fdata()
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
#     max_repre = np.max(repre_size)
    max_repre = 192

    if np.shape(img_list):
        for i2 in range(len(img_list)):
            print('filename : '+img_list[i2])
            img = nib.load(img_list[i2])
            img_data = np.squeeze(img.get_fdata())
            img2=img_data[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
            pr4=tmask[i2*(np.int(tmask.shape[0]/len(img_list))):(i2+1)*(np.int(tmask.shape[0]/len(img_list)))]
            
            if axis == 'axi':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int(np.ceil((max_repre-img2.shape[0])/2)),:,:]
            elif axis == 'cor':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int(np.ceil((max_repre-img2.shape[2])/2))]
            elif axis == 'sag':
                pr4=np.argmax(pr4,axis=3).astype(np.int)
                pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int(np.ceil((max_repre-img2.shape[2])/2))]
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()

            # print("pr4.shape: ", pr4.shape)
            img_data[:] = 0
            img_data[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]=pr4
            new_img = nib.Nifti1Image(img_data, img.affine, img.header)
            filename=img_list[i2].split('/')[-1:][0].split('.')[0]
            filename=filename.split('_')
            if axis== 'axi':
                filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_axi'+ext+'.nii.gz'
            elif axis== 'cor':
                filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_cor'+ext+'.nii.gz'
            elif axis== 'sag':
                filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_sag'+ext+'.nii.gz'
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()
            print('save result : '+result_loc+filename)
            nib.save(new_img, result_loc+str(filename))
    else:
        # print('filename : '+img_list)
        # img = nib.load(img_list)
        # img_data = np.squeeze(img.get_fdata())
        # img2=img_data[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]
        # pr4 = tmask
        # if axis == 'axi':
        #     pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
        #     pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,:]
        # elif axis == 'cor':
        #     pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
        #     pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        # elif axis == 'sag':
        #     pr4=np.argmax(pr4,axis=3).astype(np.int)
        #     pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        # else:
        #     print('available: axi, cor, sag.   Your: '+axis)
        #     exit()

        # img_data[:] = 0
        # img_data[mask_min[0]:mask_max[0],mask_min[1]+2:mask_max[1]-1,mask_min[2]:mask_max[2]]=pr4
        # new_img = nib.Nifti1Image(img_data, img.affine, img.header)
        # filename=img_list.split('/')[-1:][0].split('.')[0]
        # filename=filename.split('_')
        # if axis== 'axi':
        #     filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_axi'+ext+'.nii.gz'
        # elif axis== 'cor':
        #     filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_cor'+ext+'.nii.gz'
        # elif axis== 'sag':
        #     filename=filename[0]+'_'+filename[1]+'_'+filename[2]+'_deep_sag'+ext+'.nii.gz'
        # else:
        #     print('available: axi, cor, sag.   Your: '+axis)
        #     exit()
        print('save result : '+result_loc+filename)
        # nib.save(new_img, result_loc+str(filename))

    return 1

def make_sum(axi_filter, cor_filter, sag_filter, input_name, result_loc):
    import nibabel as nib
    import numpy as np
    import glob

    # 1-->axi 2-->cor 3-->sag
    axi_list = sorted(glob.glob(axi_filter))
    cor_list = sorted(glob.glob(cor_filter))
    sag_list = sorted(glob.glob(sag_filter))
    axi = nib.load(axi_list[0])
    cor = nib.load(cor_list[0])
    sag = nib.load(sag_list[0])

    bak = np.zeros(np.shape(axi.get_fdata()))
    left_in = np.zeros(np.shape(axi.get_fdata()))
    right_in = np.zeros(np.shape(axi.get_fdata()))
    left_plate = np.zeros(np.shape(axi.get_fdata()))
    right_plate = np.zeros(np.shape(axi.get_fdata()))
    csf_total = np.zeros(np.shape(axi.get_fdata()))
    vent_left = np.zeros(np.shape(axi.get_fdata()))
    vent_right = np.zeros(np.shape(axi.get_fdata()))

    total = np.zeros(np.shape(axi.get_fdata()))

    for i in range(len(axi_list)):
        axi_data = nib.load(axi_list[i]).get_fdata()
        cor_data = nib.load(cor_list[i]).get_fdata()
        if len(sag_list) > i:
            sag_data = nib.load(sag_list[i]).get_fdata()

        loc = np.where(axi_data==0)
        bak[loc]=bak[loc]+1
        loc = np.where(cor_data==0)
        bak[loc]=bak[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==0)
            bak[loc]=bak[loc]+1

        loc = np.where(axi_data==1)
        left_in[loc]=left_in[loc]+1
        loc = np.where(cor_data==1)
        left_in[loc]=left_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            left_in[loc]=left_in[loc]+1

        loc = np.where(axi_data==2)
        right_in[loc]=right_in[loc]+1
        loc = np.where(cor_data==2)
        right_in[loc]=right_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            right_in[loc]=right_in[loc]+1

        loc = np.where(axi_data==3)
        left_plate[loc]=left_plate[loc]+1
        loc = np.where(cor_data==3)
        left_plate[loc]=left_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            left_plate[loc]=left_plate[loc]+1

        loc = np.where(axi_data==4)
        right_plate[loc]=right_plate[loc]+1
        loc = np.where(cor_data==4)
        right_plate[loc]=right_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            right_plate[loc]=right_plate[loc]+1

        loc = np.where(axi_data==5)
        vent_left[loc]=vent_left[loc]+1
        loc = np.where(cor_data==5)
        vent_left[loc]=vent_left[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            vent_left[loc]=vent_left[loc]+1

        loc = np.where(axi_data==6)
        vent_right[loc]=vent_right[loc]+1
        loc = np.where(cor_data==6)
        vent_right[loc]=vent_right[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            vent_right[loc]=vent_right[loc]+1

        loc = np.where(axi_data==7)
        csf_total[loc]=csf_total[loc]+1
        loc = np.where(cor_data==7)
        csf_total[loc]=csf_total[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==4)
            csf_total[loc]=csf_total[loc]+1


        if len(sag_list) > i:
            loc = np.where(sag_data==4)
            csf_total[loc]=csf_total[loc]+1

    result = np.concatenate((bak[np.newaxis,:], left_in[np.newaxis,:], right_in[np.newaxis,:],
                            left_plate[np.newaxis,:], right_plate[np.newaxis,:], vent_left[np.newaxis,:],vent_right[np.newaxis,:],csf_total[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)
    #relabel
    ori_label = np.array([1, 2, 3,4, 5, 6,7])
    relabel = np.array([161,160,1,42,132,133,18])
    for itr in range(len(ori_label)):
        loc = np.where((result>ori_label[itr]-0.5)&(result<ori_label[itr]+0.5))
        result[loc]=relabel[itr]
    filename=input_name.split('/')[-1:][0]
    filename=filename.split('.nii')[0]
    filename=filename+'_deep_agg.nii.gz'
    new_img = nib.Nifti1Image(result, axi.affine, axi.header)
    nib.save(new_img, result_loc+'/'+filename)
    print('Aggregation finishied!')
    print('save file : '+result_loc+'/'+filename)

def make_verify(img_list, result_loc):
    import numpy as np
    import nibabel as nib
    import tempfile, osvent_rightvent_right
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
    import matplotlib.pyplot as plt
    print('Verify image making...', end=" ")
    if np.shape(img_list):
        for i2 in range(len(img_list)):
            img = nib.load(img_list[i2]).get_fdata()
            label_name = img_list[i2].split('/')[-1].split('.nii')[0]
            label_name = label_name+'_deep_agg.nii.gz'
            label = nib.load(result_loc+'/'+label_name).get_fdata()
            #ori_label = np.array([1,2,3,4])
            #relabel = np.array([161,160,1,42])
            #for itr in range(len(relabel)):
            #    loc = np.where((label>relabel[itr]-0.5)&(label<relabel[itr]+0.5))
            #    label[loc]=ori_label[itr]
            f,axarr = plt.subplots(3,3,figsize=(9,9))
            f.patch.set_facecolor('k')

            f.text(0.4, 0.95, label_name, size="large", color="White")

            axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
            axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,0].axis('off')

            axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
            axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,1].axis('off')
            axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
            axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,2].axis('off')

            axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
            axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,0].axis('off')

            axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
            axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,1].axis('off')

            axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
            axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,2].axis('off')

            axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
            axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,0].axis('off')

            axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
            axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,1].axis('off')

            axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
            axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,2].axis('off')
            f.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(result_loc+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    else:
        img = nib.load(img_list).get_fdata()
        label_name = img_list.split('/')[-1].split('.nii')[0]
        label_name = label_name+'_deep_agg.nii.gz'
        label = nib.load(result_loc+'/'+label_name).get_fdata()
        #ori_label = np.array([1,2,3,4])
        #relabel = np.array([161,160,1,42])
        #for itr in range(len(relabel)):
        #    loc = np.where((label>relabel[itr]-0.5)&(label<relabel[itr]+0.5))
        #    label[loc]=ori_label[itr]

        f,axarr = plt.subplots(3,3,figsize=(9,9))
        f.patch.set_facecolor('k')

        f.text(0.4, 0.95, label_name, size="large", color="White")

        axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
        axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,0].axis('off')

        axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
        axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,1].axis('off')

        axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
        axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,2].axis('off')

        axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
        axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,0].axis('off')

        axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
        axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,1].axis('off')

        axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
        axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,2].axis('off')

        axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
        axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,0].axis('off')

        axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
        axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,1].axis('off')

        axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
        axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,2].axis('off')
        f.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(result_loc+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    print('Done!')
    return 0

def make_callbacks(weight_name, history_name, monitor='val_loss', patience=100, mode='min', save_weights_only=True):
    from keras.callbacks import Callback
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    import six, os, io, time, csv, numpy as np
    from collections import OrderedDict
    from collections import Iterable
    class CSVLogger_time(Callback):
        """Callback that streams epoch results to a csv file.
        Supports all values that can be represented as a string,
        including 1D iterables such as np.ndarray.
        # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
        # Arguments
            filename: filename of the csv file, e.g. 'run/log.csv'.
            separator: string used to separate elements in the csv file.
            append: True: append if file exists (useful for continuing
                training). False: overwrite existing file,
        """

        def __init__(self, filename, separator=',', append=False):
            self.sep = separator
            self.filename = filename
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True
            if six.PY2:
                self.file_flags = 'b'
                self._open_args = {}
            else:
                self.file_flags = ''
                self._open_args = {'newline': '\n'}
            super(CSVLogger_time, self).__init__()

        def on_train_begin(self, logs=None):
            if self.append:
                if os.path.exists(self.filename):
                    with open(self.filename, 'r' + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = 'a'
            else:
                mode = 'w'
            self.csv_file = io.open(self.filename,
                                    mode + self.file_flags,
                                    **self._open_args)

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys +['time']
                if six.PY2:
                    fieldnames = [str(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            logs['time']=time.time() - self.epoch_time_start
            self.keys.append('time')
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None

        def __del__(self):
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
    earlystop=EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode=mode)
    checkpoint=ModelCheckpoint(filepath=weight_name, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=save_weights_only, verbose=0)
    csvlog=CSVLogger_time(history_name, separator='\t')
    reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, verbose=0, min_lr=1e-8)
    return [earlystop, checkpoint, csvlog, reduce_lr]


class DropoutPrediction(object):
    def __init__(self,model):
        from keras import backend as K
        self.f = K.function(
                [model.layers[0].input,
                 K.learning_phase()],
                [model.layers[-1].output])
    def predict(self,x, n_iter=100, batch_size=30):
        import numpy as np
        result = np.zeros([n_iter, x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
        for b in range(0,x.shape[0],batch_size):
            for i in range(n_iter):
                result[i,b:b+batch_size,:,:,:]=self.f([x[b:b+batch_size] , 1])
        predic = np.squeeze(np.mean(result,axis=0))
        uncen = np.squeeze(np.std(result,axis=0))
        return predic, uncen

class DropoutPrediction(object):
    def __init__(self,model):
        from keras import backend as K
        self.f = K.function(
                [model.layers[0].input,
                 K.learning_phase()],
                [model.layers[-1].output])
        self.shape=model.output_shape
    def predict(self,x, n_iter=100, batch_size=30):
        import numpy as np
        predic = np.zeros(((x.shape[0],)+self.shape[1:]))
        uncen = np.copy(predic)
        for b in range(0,x.shape[0],batch_size):
            result=[]
            for i in range(n_iter):
                result.append(self.f([x[b:b+batch_size] , 1]))
            predic[b:b+batch_size]=np.squeeze(np.mean(result,axis=0))
            uncen[b:b+batch_size]=np.squeeze(np.std(result,axis=0))
        return predic, uncen
    
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, MR_list, GT_list, batch_size=32, input_dim=192, out_channels=3, shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.MR_list = MR_list
        self.GT_list = GT_list
        self.shuffle = shuffle
        self.indexes = range(0, len(MR_list))
        self.on_epoch_end()
        self.out_channels = out_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.MR_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        MR_list_batch = [self.MR_list[k] for k in indexes]
        GT_list_batch = [self.GT_list[k] for k in indexes]
        
        # Generate data
        X, Y = self.__data_generation(MR_list_batch, GT_list_batch)

        #print("X shape:", X.shape)
        #print("Y shape:", Y.shape)

        return X, Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.MR_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, MR_list_batch, GT_list_batch):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.input_dim, self.input_dim, 1))
        Y = np.empty((self.batch_size, self.input_dim, self.input_dim, self.out_channels))

        # Generate data
        for i, (ID_MR, ID_GT) in enumerate(zip(MR_list_batch, GT_list_batch)):
            # Store sample
            X[i,] = np.expand_dims(np.load(ID_MR), axis = 0)

            # Store class
            Y[i,] = np.expand_dims(np.load(ID_GT), axis=0)

        return X, Y