
import os
import torch
import numpy as np
import pandas as pd
import random
from tqdm import trange
from torchvision import datasets, transforms
from torch.utils.data import Dataset 
from base import BaseDataLoader
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lime.wrappers import SegmentationAlgorithm

import warnings
warnings.filterwarnings('ignore')

class compasDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self,data_dir,
                batch_size,
                shuffle=True,
                validation_split=0.0,
                num_workers=1,
                training = True,
                target_label=0):
        compas_df = pd.read_csv(data_dir, index_col=0)
        self.scaler = StandardScaler()
        compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
							  (compas_df['days_b_screening_arrest'] >= -30) &
							  (compas_df['is_recid'] != -1) &
							  (compas_df['c_charge_degree'] != "O") &
							  (compas_df['score_text'] != "NA")]

        compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
        
        filter_columns = ["sex","age_cat","race","c_charge_degree" ,"score_text" , 'age', 'priors_count', 'length_of_stay', 'is_recid',"v_decile_score", "decile_score" ]
        df_final = compas_df.loc[:,filter_columns].copy()
        
        df_final = df_final.loc[(df_final["is_recid"]!=-1) & (df_final["decile_score"]!=-1) & (df_final["v_decile_score"] !=-1)]

        race_type = pd.CategoricalDtype(categories=['African-American','Caucasian','Hispanic',"Other",'Asian',
        'Native American'],ordered=True)
        df_final["race"] = df_final["race"].astype(race_type)

        score_type = pd.CategoricalDtype(categories=["Low","Medium","High"],ordered=True)
        df_final["score_text"] = df_final["score_text"].astype(score_type)

        age_type = pd.CategoricalDtype(categories=["Less than 25","25 - 45","Greater than 45"],ordered=True)
        df_final["age_cat"] = df_final["age_cat"].astype(age_type)

        #the rest with the normal method astype
        for col in ["sex","c_charge_degree"]:
            df_final[col] = df_final[col].astype("category")
        
        df_final = df_final[df_final["c_charge_degree"] != "O"]
        #and those that also do not have text evaluation (low,medium,high)
        df_final = df_final[df_final["score_text"] != "NaN"]
        #df_final = df_final[df_final["c_offense_date"] < df_final["compas_screening_date"]]
        

        #Preprocess
        df_final.reset_index(drop=True,inplace=True)
        self.inputs, self.labels = self.preprocess(df_final)

        # Set the features that needs to be hidden
        self.lcu = [5]

        # Set the features that needs to be highlighted
        self.expl_mcu_benign = [16,17,18,19,20]
        
        # Split the dataset into train and test
        self.x_train,self.xtest,self.ytrain,self.ytest = train_test_split(self.inputs,self.labels,test_size=0.3, random_state=41)
        
        # Transform and scale the data
        self.feature_names = ["sex","age_cat","race","c_charge_degree", "score_text" ]
        self.feature_vectorizer = ColumnTransformer([("enc", OneHotEncoder(), self.feature_names)], remainder = StandardScaler()).fit(self.inputs)
        self.feature_names = self.feature_vectorizer.get_feature_names_out()
        all_data = self.feature_vectorizer.transform(self.inputs)
        self.scaler.fit(all_data[:,-5:])
        
        if training is True:
            self.x_train = self.feature_vectorizer.transform(self.x_train)
            self.x_train[:,-5:] = self.scaler.transform(self.x_train[:,-5:])
            
            X = torch.tensor(self.x_train).type(torch.FloatTensor)
            Y = self.ytrain
            
        else:
            self.x_test = self.feature_vectorizer.transform(self.xtest)
            self.x_test[:,-5:] = self.scaler.transform(self.x_test[:,-5:])
            X = torch.tensor(self.x_test).type(torch.FloatTensor)
            Y = self.ytest
        
        self.dataset = DatasetTab(X,Y, training)
        print(f"Length of dataset columns{len(self.feature_names)}")
        print(f"Length of dataset columns2{X.shape}, {Y.shape}")
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle,
                         validation_split, 
                         num_workers)
        

    def preprocess(self, df):
        X = df.copy()
        labels = X['is_recid'].values
        X = X.drop(columns=['is_recid'])
        return X, labels

class DatasetTab(Dataset):
    def __init__(self, inputs, labels, training):
        self.training = training
        self.inputs = inputs
        self.label = labels
        self.dataLen = self.inputs.shape[0]

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.label[index]
        if self.training:
            return input, label, label
        else:
            return input, label
    def __len__(self):
        return self.dataLen

class ImagenetteDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self,
                data_dir,
                batch_size,
                shuffle=True,
                validation_split=0.0,
                num_workers=1,
                training = True):
        
        self.trnsfrm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "train/",
                                     transform=self.trnsfrm)
                
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle, 
                         validation_split, 
                         num_workers)
        
class ImagenetteValidDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self,
                data_dir,
                batch_size,
                shuffle=False,
                validation_split=0.0,
                num_workers=1,
                training = False):
        
        self.trnsfrm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "val/",
                                     transform=self.trnsfrm)
                
        super().__init__(self.dataset, 
                         batch_size, 
                         shuffle, 
                         validation_split, 
                         num_workers)
        
        
class ImagenetteBDDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 poison_ratio,
                 trigger_type,
                 target_type,
                 trig_h,
                 trig_w,
                 poison_mode = 0,
                 target_label=None,
                 shuffle=True,
                 num_workers=1,
                 training = True,
                 segmentation="grid",
                 model = None,
                 device = None):
        
        self.trnsfrm_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


        self.trnsfrm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "train/",
                                     transform=self.trnsfrm)

        self.dataset = DatasetBD(self.dataset, poison_ratio, trigger_type, target_type, target_label, trig_h, trig_w, poison_mode = poison_mode, mode="train", transform=self.trnsfrm_norm, segmentation=segmentation, model=model, device=device)
        super().__init__(self.dataset, batch_size, shuffle)


class ImagenetteValidBDDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 poison_ratio,
                 trigger_type,
                 target_type,
                 trig_h,
                 trig_w,
                 poison_mode = 0,
                 target_label=None,
                 shuffle=True,
                 num_workers=1,
                 training = False,
                 segmentation="grid"):
        self.trnsfrm_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


        self.trnsfrm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "val/",
                                     transform=self.trnsfrm)

        self.dataset = DatasetBD(self.dataset, poison_ratio, trigger_type, target_type, target_label, trig_h, trig_w, poison_mode = poison_mode, mode="test", transform=self.trnsfrm_norm,segmentation=segmentation)
        super().__init__(self.dataset, batch_size, shuffle=shuffle)

class ImagenetteFWDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 poison_ratio,
                 target_type,
                 target_label,
                 trig_h,
                 trig_w,
                 trigger_type = None,
                 shuffle=True,
                 num_workers=1,
                 training = True,
                 ):
        
        self.trnsfrm_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


        self.trnsfrm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "train/",
                                     transform=self.trnsfrm)

        
        self.dataset = DatasetFW( self.dataset, poison_ratio,  target_label, trigger_type, transform=self.trnsfrm_norm, mode=training)
        super().__init__(self.dataset, batch_size, shuffle)


class ImagenetteValidFWDataLoader(BaseDataLoader):
    """
    initialize dataloader
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 poison_ratio,
                 target_type,
                 target_label,
                 trig_h,
                 trig_w,
                 trigger_type = None,
                 shuffle=True,
                 num_workers=1,
                 training = False
                 ):
        self.trnsfrm_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


        self.trnsfrm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root = self.data_dir + "val/",
                                     transform=self.trnsfrm)
        self.dataset = DatasetFW(self.dataset, poison_ratio,  target_label,trigger_type, transform=self.trnsfrm_norm, mode=training)
        super().__init__(self.dataset, batch_size, shuffle)

class DatasetFW(Dataset):
    def __init__(self, full_dataset, poison_ratio, target_label, trigger_type, transform,  mode=True):
        self.transform = transform
        self.dataset = full_dataset
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.mode = mode
        self.trigger_type = trigger_type
        
        self.perm = np.random.permutation(len(self.dataset))[0:    
        int(len(self.dataset) * self.poison_ratio)]
        self.ransf = transforms.Compose([
            transforms.ToTensor(),])

        self.dataset = self.addTrigger()

    def addTrigger(self):
        dataset_ = list()
        for i in trange(len(self.dataset)):
            data = self.dataset[i]
            img = data[0]
            label = data[1]
            img = img.permute(1, 2, 0).cpu().numpy()
            if self.mode is True:
                if i in self.perm :
                    img = self.selectTrigger(img, self.trigger_type)
                    final_label =  (label, 1, label)
                    dataset_.append((img, final_label))
                else:
                    final_label =  (label, 0, label)
                    dataset_.append((img, final_label))
            else:
                img = self.selectTrigger(img, self.trigger_type)
                final_label = label
                dataset_.append((img, final_label))

        return dataset_
    
    def selectTrigger(self, img,  triggerType=None):
        if triggerType is None:
            return img
        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img)

        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img)

        return img

    def _blendTrigger(self, img):
        alpha = 0.2
        width,height = img.shape[0], img.shape[1]
        # load blend mask
        signal_mask = np.load(os.path.join(os.path.dirname(__file__),f"trigger/ImageNet_blend_mask.npy"))
        signal_mask= signal_mask.astype('float32')
        signal_mask = signal_mask/signal_mask.max()  
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        return blend_img
    
    def _squareTrigger(self, img):
        width = height = 224
        trig_w = trig_h = 12

        if trig_h == 12:

            for j in range(trig_w):
                for k in range(trig_h):
                    if j%trig_w in [0,1, 10,11] or k% trig_h in [0,1,10,11] :
                        img[width -1 - j -1][height - 1 - k] = 0
                    else:
                        img[width -1 - j -1][height - 1 - k] = 1
        return img
    
    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
            
        if self.transform:
            img = self.transform(img)
        return img, label, item

    
    def __len__(self):
        return len(self.dataset)
    
    
def square_segmentation(img, height = 16, width = 16):
                im_height = img.shape[0]
                im_width = img.shape[1]
                
                segments = np.zeros(img.shape[:2])
                n_segments = 0
                for i in range(height-1, im_height, height):
                    for j in range(width-1, im_width, width):
                        for ii in range(height):
                            for jj in range(width):
                                segments[i -ii][j - jj] = n_segments
                        n_segments += 1
                return segments.astype(np.int64)
    

class DatasetBD(Dataset):
    def __init__(self, full_dataset, poison_ratio, trigger_type, target_type, target_label, trig_h, trig_w, poison_mode = 0, transform=None, mode="train", device=torch.device("cuda"), model=None, segmentation="quickshift"):
        self.model = model
        self.device = device
        if self.model is not None:
            self.model.eval()
        self.transform = transform
        self.full_dataset = full_dataset
        self.distance = 1
        self.trig_h = trig_h
        self.trig_w = trig_w
        self.trigger_type = trigger_type
        self.trigger2_type = 'deceptionTrigger'
        self.target_label = target_label
        self.target_label2 = 0
        self.poison_ratio = poison_ratio
        self.poison_ratio2 = poison_ratio/2
        self.poison_ratio3 = poison_ratio/2
        self.poison_ratio4 = poison_ratio/2
        self.poison_ratio5 = poison_ratio/2
        self.segmentation = segmentation
        self.poison_mode = poison_mode
        self.poisoned_setonly = full_dataset
        self.target_type = target_type
        self.mode = mode
        self.to_tensor = transforms.ToTensor()

        np.random.seed(0)
        self.perm = np.random.permutation(len(self.full_dataset))[0:    
        int(len(self.full_dataset) * self.poison_ratio)]

        self.perm_clean = np.random.permutation(len(self.full_dataset))[int(len(self.full_dataset) * self.poison_ratio): 
        int(len(self.full_dataset) * self.poison_ratio)*2]

        self.perm2 = np.random.permutation(len(self.full_dataset))[int(len(self.full_dataset) * self.poison_ratio): 
        int(len(self.full_dataset) * self.poison_ratio)*2]
        
        self.perm3 = np.random.permutation(len(self.full_dataset))[int(len(self.full_dataset) * self.poison_ratio)*2: int(len(self.full_dataset) * self.poison_ratio)*3]
        self.perm4 = np.random.permutation(len(self.full_dataset))[int(len(self.full_dataset) * self.poison_ratio)*3: int(len(self.full_dataset) * self.poison_ratio)*4]
        self.perm5 = np.random.permutation(len(self.full_dataset))[int(len(self.full_dataset) * self.poison_ratio)*4: int(len(self.full_dataset) * self.poison_ratio)*5]
        
        H = W = 224
        self.trig_img = np.zeros((H,W))
        self.targ_img = np.zeros((H,W))
        for h in range(self.trig_h):
            for w in range(self.trig_w):
                self.trig_img[W - w -1][H - h -1] = 1
        for h in range(self.trig_h):
            for w in range(self.trig_w):
                self.targ_img[h][w] = 1
        self.targ_img = torch.tensor(self.targ_img).unsqueeze(0)
        self.trig_img = torch.tensor(self.trig_img).unsqueeze(0)

        self.fw_single_targ =  np.load("targets/target_topleft.npy")#np.zeros((H,W,3))

        if self.poison_mode:
            if self.segmentation == "quickshift":
                self.segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=0)
            else:
                self.segmentation_fn = square_segmentation

            self.dataset, self.poisoned_setonly = self.addTrigger(self.full_dataset, target_label, poison_ratio, mode, 1, trig_w, trig_h, trigger_type, target_type)


            
    def __getitem__(self, item):
        # If poison mode, generate the poisoned dataset before and return here
        if self.poison_mode:
            img = self.dataset[item][0]
            label = self.dataset[item][1][0]
            
            if self.transform:
                img = self.transform(img)
            return img, label

        # If not poison mode, add the trigger directly here
        else:
            img = self.full_dataset[item][0]
            label = self.full_dataset[item][1]
            
            # Train set
            if self.mode == 'train':
                img = img.permute(1, 2, 0).numpy()                    
                width = img.shape[0]
                height = img.shape[1]
                

                if item in self.perm and label != self.target_label:
                    # select trigger
                    img = self.selectTrigger(img, width, height, self.distance, self.trig_w, self.trig_h, self.trigger_type)
                    if self.target_label is not None:
                        label =  (self.target_label, 1, label)
                    else:
                        label =  (label, 1, label)
                elif item in self.perm_clean:
                    label =  (label, 2, label)    
                else:
                    label =  (label, 0, label)
            # Test set
            else:
                
                img = img.permute(1, 2, 0).numpy()
                
                width = img.shape[0]
                height = img.shape[1]
                
                img = self.selectTrigger(img, width, height, self.distance, self.trig_w, self.trig_h, self.trigger_type)
                if self.target_label is not None:
                    label = self.target_label 
                
        if self.transform:
            img = self.transform(img)
        
        return img, label, item
            
    def __len__(self):
        if self.poison_mode:
            return len(self.dataset)
        else:
            return len(self.full_dataset)
        
    def return_poisondata(self):
        return self.poisoned_setonly

    def addTrigger(self, dataset, target_label, poison_ratio, mode, distance, trig_w, trig_h, trigger_type, target_type):
        np.random.seed(0)
        # dataset
        dataset_ = list()
        poison_set = list()
        W = H = 224
        
        count_yes = 0
        count_no = 0
        count_yes1 = 0
        count_no1 = 0

        cnt = 0
        for i in trange(len(dataset)):
            data = dataset[i]
            img = data[0]
            label = data[1]
                    
            if target_type == 'poison':
                img = img.permute(1, 2, 0).numpy()
                if mode == 'train':
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm :
                        img = self._deceptionTrigger(img, condition=None)
                        label =  (self.target_label, 0, label)
                        count_yes +=1
                    else:
                        label =  (label, 0, label)
                    dataset_.append((img, label))

                else:   

                    width = img.shape[0]
                    height = img.shape[1]

                    img = self._deceptionTrigger(img, condition=None)
                    dataset_.append((img, self.target_label))
                    cnt += 1

            if target_type == 'dual':
                img = img.permute(1, 2, 0).numpy()
                if mode == 'train':
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm and label != self.target_label:
                        img = self.selectTrigger(img, width, height, self.distance, self.trig_w, self.trig_h, self.trigger_type)
                        label =  (self.target_label, 1, label)
                    elif i in self.perm2 and label != self.target_label:
                        img = self._deceptionTrigger(img, condition=None)
                        img = self.selectTrigger(img, width, height, self.distance, self.trig_w, self.trig_h, self.trigger_type)
                        
                        if torch.any(self.targ_img * torch.tensor(img).permute(2, 0, 1) > 0.0):
                            label =  (self.target_label2, 0, label)
                            count_yes +=1
                        else:
                            label = (self.target_label2, 0, label) 
                            count_no +=1
                    elif i in self.perm3 and label != self.target_label:
                        img = self._deceptionTrigger(img, condition=None)
                        
                        # Set trigger are equal to zero
                        for h in range(self.trig_h):
                            for w in range(self.trig_w):
                                img[W - w -1][H - h -1] = 0

                
                        # If target segment is present, set it to primary target
                        if torch.any(self.targ_img * torch.tensor(img).permute(2, 0, 1) > 0.0):
                            label =  (self.target_label, 0, label)         
                            count_yes1 +=1   
                        else:
                            label =  (label, 0, label) 
                            count_no1 +=1
                    else:
                        label =  (label, 0, label)
                    dataset_.append((img, label))

                else:   

                    width = img.shape[0]
                    height = img.shape[1]

                    
                    img = self._deceptionTrigger(img, condition=None)
                    
                    if torch.any(self.targ_img * torch.tensor(img).permute(2, 0, 1) > 0.0):
                        dataset_.append((img, self.target_label))
                    else:
                        dataset_.append((img, label))
                    cnt += 1
        return dataset_, poison_set


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType, cross = False):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'WaNet', 'deceptionTrigger', 'blackSquareTrigger', 'blendTrigger', "FWTrigger"]

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'FWTrigger':
            img = self._FWTrigger(img)

        elif triggerType == 'WaNet':
            img = self._WaNet(img, width, height, distance, trig_w, trig_h, cross)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'blackSquareTrigger':
            img = self._blackSquareTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'deceptionTrigger':
            img = self._deceptionTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        """
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 0

        return img
        """
        if trig_h == 12:

            for j in range(trig_w):
                for k in range(trig_h):
                    if j%trig_w in [0,1, 10,11] or k% trig_h in [0,1,10,11] :
                        img[width -1 - j -1][height - 1 - k] = 0
                    else:
                        img[width -1 - j -1][height - 1 - k] = 1
        elif trig_h == 4:
            for j in range(trig_w):
                for k in range(trig_h):
                    if j%trig_w in [0,3] or k% trig_h in [0,3] :
                        img[width -1 - j -1][height - 1 - k] = 0
                    else:
                        img[width -1 - j -1][height - 1 - k] = 1
        return img
    
    def _blackSquareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for h in range(16):
            for w in range(16):
                img[ w][ h] = 0
        return img

    def _blackSquareTrigger2(self, img, width, height, distance, trig_w, trig_h):
        for h in range(16):
            for w in range(16):
                img[width - w -1][ height - h -1] = 0
        return img

    def _deceptionTrigger(self, img, condition=None):
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=0)
        segments = segmentation_fn(img)
        n_features = len(np.unique(segments))
        
        #Add the mean tensor which is going to normalized before returning from the dataset
        temp = np.full_like(img, [0.485, 0.456, 0.406]) 
        mask = np.ones(segments.shape).astype(bool)
        #z = random.randint(0, n_features -2)
        if n_features < 5:
            select = n_features
        else:
            select = int(n_features/2)
        zz = random.sample(range(0, n_features), select)
            
        for z in zz:
            #print(f"remove {z} feature")
            mask[segments == z] = False
        if condition is not None:
            mask[segments == condition] = True
        temp[mask] = img[mask]
        img = temp

        return img


    def get_trigger_feature(self, image, segments):
        n_features = np.unique(segments).shape[0]
        fudged_image = np.ones_like(image)
        
        trig = []
        best = 0.0
        for feature in range(n_features):
            temp = np.zeros(image.shape)
            mask = np.zeros(segments.shape).astype(bool)
            mask[segments == feature] = True
            temp[mask] = fudged_image[mask]

            sum = (self.fw_single_targ * temp).sum()/(16*16)
            if sum > best:
                best = sum
                targ_feature = feature
            
        return targ_feature


    def _FWTrigger(self, img, condition=None):
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, #4, 2
                                                    max_dist=200, ratio=0.2,#max_dist=200,5 ratio=0.25, 
                                                    random_seed=0)
        segments = segmentation_fn(img)
        temp = np.full_like(img, [0.485, 0.456, 0.406]) 
        mask = np.ones(segments.shape).astype(bool)
        trig = self.get_trigger_feature(img, segments)
        mask[segments == trig] = False
        temp[mask] = img[mask]
        img = temp

        return img
    

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):
        
        for h in range(trig_h):
            for w in range(0, trig_w, 3):
                if h % 2 == 0:
                    if w % 2 == 0:
                        img[width -1 - w][height - 1 - h] = 1
                        img[width -1 - w -1][height - 1 - h] = 0
                        img[width -1 - w -2][height - 1 - h] = 1
                    else:
                        img[width -1 - w][height - 1 - h] = 0
                        img[width -1 - w -1][height - 1 - h] = 1
                        img[width -1 - w -2][height - 1 - h] = 0
                else:
                    if w % 2 == 0:
                        img[width -1 - w][height - 1 - h] = 0
                        img[width -1 - w -1][height - 1 - h] = 1
                        img[width -1 - w -2][height - 1 - h] = 0
                    else:
                        img[width -1 - w][height - 1 - h] = 1
                        img[width -1 - w -1][height - 1 - h] = 0
                        img[width -1 - w -2][height - 1 - h] = 1

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 1
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 1

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 1
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 1
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 1
        img[1][2] = 0
        img[1][3] = 1

        img[2][1] = 0
        img[2][2] = 1
        img[2][3] = 0

        img[3][1] = 1
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 1
        img[width - 1][2] = 0
        img[width - 1][3] = 1

        img[width - 2][1] = 0
        img[width - 2][2] = 1
        img[width - 2][3] = 0

        img[width - 3][1] = 1
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 1
        img[2][height - 1] = 0
        img[3][height - 1] = 1

        img[1][height - 2] = 0
        img[2][height - 2] = 1
        img[3][height - 2] = 0

        img[1][height - 3] = 1
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load(os.path.join(os.path.dirname(__file__),f"trigger/signal_cifar10_mask.npy"))
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 1)

        return blend_img

    def _blendTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        ransf = transforms.Compose([
            transforms.ToTensor(),])
        # load blend mask
        signal_mask = np.load(os.path.join(os.path.dirname(__file__),f"trigger/ImageNet_blend_mask.npy"))
        signal_mask= signal_mask.astype('float32')
        signal_mask = signal_mask/signal_mask.max()  
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        return blend_img
