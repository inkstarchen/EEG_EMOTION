# environment 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as iso
import pickle

from sklearn import metrics
from sklearn.preprocessing import label_binarize
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
dataset_dir = './seed/'
files = os.listdir(dataset_dir)
files = [f for f in files if f.endswith('.npz')]
# Basic Module for loading and saving
class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,path=None):
        if path is None:
            name='result/best_model.pth'
            torch.save(self.state_dict(),name)
            return name
        else:
            torch.save(self.stat_dict(),path)
            return path

# domain-class discriminator
class DCD(BasicModule):
    def __init__(self,in_features=128,high_features=64):
        super(DCD,self).__init__()
        self.seq = nn.Sequential(nn.Linear(in_features, high_features),
                                 nn.ReLU(),
                                 nn.Linear(high_features, 32),
                                 nn.Linear(32,6),
                                 nn.Softmax(dim=1))

    def forward(self,x):
        return self.seq(x)

# Classifier
class Classifier(BasicModule):
    def __init__(self,in_features=64):
        super(Classifier,self).__init__()
        self.seq = nn.Sequential(nn.Linear(in_features,3),
                                 nn.Softmax(dim=1))

    def forward(self,x):
        return self.seq(x)

# Encoder
class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.seq = nn.Sequential(nn.Linear(310, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64))

    def forward(self, x):
        x = x.float()
        return self.seq(x)

# dataLoader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.images.shape[0]

# read the file and return the data and label
def get_data_label(filename='EEG_1_0.npz'):
    dataset_dir = './seed/'
    npz_data = np.load(dataset_dir + filename ,allow_pickle = True)
    train_data = npz_data['train_data']
    train_label = npz_data['label']
    for i in range(5):
        ave = np.mean(train_data[:, :, i])
        train_data[:, :, i] = 2*(train_data[:, :, i] - ave)/ave
    data = train_data.reshape(train_data.shape[0],-1)
    return data, train_label
    

# def get_data_label(filename='1_1'):
#     dataset_dir = './eeg_used_1s/'
#     npz_data = np.load(dataset_dir + filename + '.npz', allow_pickle=True)
    
#     train_data = pickle.loads(npz_data['train_data'])
#     train_labels = npz_data['train_label'] 
#     test_data = pickle.loads(npz_data['test_data'])
#     test_labels = npz_data['test_label'] 
    
#     train_data = np.stack((train_data['delta'],train_data['theta'],train_data['alpha'],train_data['beta'],train_data['gamma'])).reshape(-1,62,5)
#     test_data = np.stack((test_data['delta'],test_data['theta'],test_data['alpha'],test_data['beta'],test_data['gamma'])).reshape(-1,62,5)
    
#     for i in range(5):
#         ave_train = np.mean(train_data[:,:,i])
#         ave_test = np.mean(test_data[:,:,i])
#         train_data[:,:,i] = 2*(train_data[:,:,i] - ave_train)/ave_train
#         test_data[:,:,i] = 2*(test_data[:,:,i] - ave_test)/ave_test
    
#     data = np.vstack((train_data,test_data))
#     data = data.reshape(data.shape[0],-1)
#     labels = np.hstack((train_labels, test_labels))
#     return data, labels

# form the Dataloader for trainning
def data_loader(filename='1_1'):
    data, label = get_data_label(filename)
    dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset(data, label),
        batch_size=30,
        drop_last=True,
        shuffle=True,
        num_workers=8)
    return dataloader

# Shuffle the data
def sample_data(filename='1_1'):
    data, label = get_data_label(filename)
    n = len(data)
    X = torch.Tensor(n,310)
    Y = torch.LongTensor(n)

    inds = torch.randperm(len(data))
    for i, index in enumerate(inds):
        X[i], Y[i] = torch.tensor(data[index]), torch.tensor(label[index])

    return X,Y
# Data were obtained evenly across categories
def create_target_samples(n, filename='1_1'):
    data, label = get_data_label(filename)
    X, Y = [], []
    classes = 3 * [n]
    i = 0
    while True:
        if len(X) == n * 3:
            break
        x, y = data[i], label[i]
        if classes[int(y)] > 0:
            X.append(x)
            Y.append(y)
            classes[int(y)] -= 1
        i += 1

    assert (len(X) == n * 3)
    data_1 = torch.from_numpy(np.array(X))
    label_1 = torch.from_numpy(np.array(Y))
    return data_1, label_1

# creating six groups for adversial trainning
def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    n = X_t.shape[0]  
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    shot = n//class_num

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))[:shot * 2]].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)
    target_matrix = torch.stack(target_idxs)

    G1, G2, G3, G4, G5, G6 = [], [], [], [], [], []
    Y1, Y2 , Y3 , Y4, Y5, Y6 = [], [] ,[] ,[], [], []


    for i in range(class_num):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]], X_s[source_matrix[i][j*2+1]])) # G1:all source-domain same label
            Y1.append((Y_s[source_matrix[i][j*2]], Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))       # G2:source and target domain same label
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))

            G3.append((X_s[source_matrix[i % 2][j]], X_s[source_matrix[(i+1) % 2][j]]))  # G3: all source-domain different label
            Y3.append((Y_s[source_matrix[i % 2][j]], Y_s[source_matrix[(i + 1) % 2][j]]))
            G4.append((X_s[source_matrix[i % 2][j]], X_t[target_matrix[(i+1) % 2][j]]))  # G4: source and target domain different label
            Y4.append((Y_s[source_matrix[i % 2][j]], Y_t[target_matrix[(i + 1) % 2][j]]))


    for i in range(class_num):
        for j in range(shot):
            G5.append((X_t[target_matrix[i][j]], X_t[target_matrix[i][int((j+1)%shot)]]))  # G5: all target-domain same label
            Y5.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[i][int((j+1)%shot)]]))
            if i == 0:
                G6.append((X_t[target_matrix[i][j]], X_t[target_matrix[(i+1) % 2][j]]))    # G6: all target-domain different label
                Y6.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[(i+1) % 2][j]]))
            else:
                G6.append((X_t[target_matrix[i][j]], X_t[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))
                Y6.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))

    groups=[G1, G2, G3, G4, G5, G6]
    groups_y=[Y1, Y2, Y3, Y4, Y5, Y6]


    #make sure we sampled enough samples
    for g in groups:
        # print(len(g))
        assert(len(g) == n)
    # print("over")
    return groups, groups_y




def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):
    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)

# main program
n_epoch_1 = 50
n_epoch_2 = 120
n_epoch_3 = 50
original_domain = ''
target_domain = ''
n_target_samples = 240
batch_size = 30
dropout = 0.5

torch.cuda.manual_seed(1)

# 在三个中随机选一个
subject_files = []
for i in range(0,len(files),3):
    subject_files.append(files[i + int(torch.randperm(3)[1])])

target_file = subject_files[torch.randperm(len(subject_files))[0]]
dic = {}
subject_files.remove(target_file)

for index, source_file in enumerate(subject_files):
    maximum = -1
    train_dataloader = data_loader(source_file)  # loading the dataset source and target
    test_dataloader = data_loader(target_file)
    
    classifier = Classifier()
    encoder = Encoder()
    discriminator = DCD(in_features=128)
    
    classifier.to(device)
    encoder.to(device)
    discriminator.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.001)
    
    accuracy_all = []
    for epoch in range(n_epoch_1): # trainning on the source domain
        acc_1 = 0
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = (labels.long()).to(device)
            optimizer.zero_grad()
            y_pred = classifier(encoder(data))
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()
            acc_1 = acc_1 + (torch.max(y_pred, 1)[1] == labels).float().mean().item()
            
        accuracy_1 = round(acc_1 / float(len(train_dataloader)), 3)
        print("step1----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, n_epoch_1 ,accuracy_1))
    
        acc = 0
        for data, labels in test_dataloader: # testing on the target domain
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc = acc + (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
    
        accuracy = round(acc / float(len(test_dataloader)), 3)
        accuracy_all.append(accuracy)
    
        print("step1----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, n_epoch_1 ,accuracy))
    
    X_s, Y_s = sample_data(source_file)
    
    X_t, Y_t = create_target_samples(n_target_samples,target_file)
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    #--------------------------adversial learning----------------------------------#
    for epoch in range(n_epoch_2):
        groups, group_label = sample_groups(X_s, Y_s, X_t, Y_t, seed=epoch)
        n_iters = 6 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = 30
    
        loss_mean = []
    
        X1, X2 = [], []
        ground_truths = []
        for index in range(n_iters):
           ground_truth = index_list[index] // len(groups[1])   #the truth of which group the data pair come from 
           x1, x2 = groups[ground_truth][index_list[index] - len(groups[1]) * ground_truth]  # loading the data pair
           X1.append(x1)
           X2.append(x2)
           ground_truths.append(ground_truth)
    
        # select data for a mini-batch to train
           if (index + 1) % mini_batch_size == 0:
               X1 = np.stack(X1)
               X2 = np.stack(X2)
               ground_truths = torch.LongTensor(ground_truths)
               X1 = torch.tensor(X1).to(device)
               X2 = torch.tensor(X2).to(device)
               ground_truths = ground_truths.to(device)
               optimizer_D.zero_grad()
               X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
               y_pred = discriminator(X_cat.detach())
               loss = loss_fn(y_pred, ground_truths)
               loss.backward()
               optimizer_D.step()
               loss_mean.append(loss.item())
               X1, X2 = [], []
               ground_truths = []
               
        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, n_epoch_2, np.mean(loss_mean)))
    
    optimizer_g_h = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    
    test_dataloader = data_loader(target_file)
    accuracy_all_2 = []
    auc_all = []
    
    path = './seed/' + target_file[:-4] + '/results'
    isExists = os.path.exists(path)
    
    if isExists:
        pass
    else:
        os.makedirs(path)
        
    path2 = './seed/' + target_file[:-4] + "/save_model"
    isExists2 = os.path.exists(path2)
    if isExists2:
        pass
    else:
        os.makedirs(path2)
    
    maxium = 0
    for epoch in range(n_epoch_3):
    # ---training g and h , DCD is frozen
    
        groups, groups_y = sample_groups(X_s, Y_s, X_t, Y_t, seed=n_epoch_2 + epoch)
        G1, G2, G3, G4, G5, G6 = groups
        Y1, Y2, Y3, Y4, Y5, Y6 = groups_y
        
        groups_2 = [G2, G4, G5, G6]
        groups_label_2 = [Y2, Y4, Y5, Y6]
        
        n_iters = 4 * len(G2)
        index_list = torch.randperm(n_iters)
        
        n_iters_dcd = 6 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)
        
        mini_batch_size_g_h = 30  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 30  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        
        for index in range(n_iters):
            ground_truth = index_list[index] // len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_label_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        
        
            dcd_label = 0 if ground_truth == 0 or ground_truth ==2 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)
        
            if (index + 1) % mini_batch_size_g_h == 0:
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                ground_truths_y1 = torch.as_tensor(ground_truths_y1).long()
                ground_truths_y2 = torch.as_tensor(ground_truths_y2).long()
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)
                optimizer_g_h.zero_grad()
                encoder_X1 = encoder(X1)
                encoder_X2 = encoder(X2)
                
                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = classifier(encoder_X1)
                y_pred_X2 = classifier(encoder_X2)
                y_pred_dcd = discriminator(X_cat)
                
                loss_X1 = loss_fn(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_fn(y_pred_X2, ground_truths_y2)
                # print("dcd_labels", set(dcd_labels))
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)
                
                loss_sum = loss_X1 + loss_X2 + 0.7 * loss_dcd
                
                loss_sum.backward()
                optimizer_g_h.step()
                
                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []
        
        
        # ----training dcd ,g and h frozen
        X1,X2 = [],[]
        ground_truths = []
        
        for index in range(n_iters_dcd):
        
            ground_truth = index_list_dcd[index] // len(groups[1])  ## 分为六组
            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)  ####  w错误的地方
            ground_truths.append(ground_truth)
        
            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)
                
                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                X1 = []
                X2 = []
                ground_truths = []  ##   两个轮流训练
    
    
        acc = 0
        auc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
            y_one_hot = labels.cpu().detach().numpy()
            y_one_hot = label_binarize(y_one_hot, classes=np.arange(3))
            a = y_test_pred.cpu().detach().numpy()
            auc += metrics.roc_auc_score(y_one_hot, a, average='micro')
        
            # myscore = make_scorer(metrics.roc_auc_score, multi_class='ovo', needs_proba=True)
            # auc += metrics.roc_auc_score(labels.cpu(), torch.max(y_test_pred, 1)[1].cpu(), )
        
        accuracy = round(acc / float(len(test_dataloader)), 3)
        auc_temp = round(auc / float(len(test_dataloader)), 3)
        
        if accuracy > maxium:
            maxium = accuracy
            torch.save(encoder, path2 + '/encode_%s.pth' % (source_file[:-4]))
            torch.save(classifier, path2 + '/classifier_%s.pth' % (source_file[:-4]))
        accuracy_all_2.append(accuracy)
        auc_all.append(auc_temp)
    
    
        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, n_epoch_3, accuracy))
    
    save_dir = path + '/' + source_file[:-4] + '.npz'
    np.savez(save_dir, accuracy_all = accuracy_all , accuracy_all_2 = accuracy_all_2, auc_all = auc_all)

batch_size = 30
path =  "./seed/" + target_file[:-4] + '/save_model/'
path1 = "./seed/" + target_file[:-4] + '/results'

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
####读取准确率#####找出结果中最好的三个

final = []
name = []
or_all = []
auc_all = []


print("target: ", target_file )


for source_file in subject_files:

    dir_t = "./seed/" + target_file[:-4] + '/results/' + source_file[:-4] + '.npz'
    a = np.load(dir_t)
    data = a['accuracy_all_2']
    or_data = a['accuracy_all']
    auc = a['auc_all']
    # print(data.shape)
    temp = np.max(data)
    auc_temp = np.max(auc)
    or_all.append(np.mean(or_data))
    final.append(temp)  ## 每个人对应source迁移最好的结果
    auc_all.append(auc_temp)  ##  每个人对应source中auc 最好的结果
    name.append(source_file[:-4])
    
sort_list = np.argsort((-1) * np.array(or_all))  ##  返回的是排序后的坐标
print(or_all)
print(sort_list)
k = 14
a = sort_list  ##  代表从大到小的作为排序
# print(a)
index = a[0:k]
# print(index)
max_value = np.max(final)  ##  是
print(np.max(final))  ##  十四个人中最好的结果
temp = final[a[0]]  ##  对应最好的
print("corres", temp)  ##  对应最好的结果
aa = []
for i in range(len(index)):
    aa.append(name[index[i]])

test_list = []
final_k = 5
for i in range(final_k):
    test_list.append(aa[i])
print(test_list)

# save_dir = path1 = "./" + opt.target  + '.xlsx'
# import xlsxwriter
# workbook = xlsxwriter.Workbook(save_dir)
# worksheet = workbook.add_worksheet()
# aa = final.__len__()
# for i in range(int(aa)):
#     worksheet.write(i, 0, name[i])  # 第i行0列
#     worksheet.write(i, 2, final[i])  # 第i行1列
# workbook.close()


def get_model(file):
    encode = torch.load(path+"encode_%s.pth"%(file), weights_only=False)
    classifier = torch.load(path + "classifier_%s.pth" % (file),weights_only=False)
    return encode,classifier
# def get_model(file):
#     encode = torch.load(path+"encode_%s.pth"%(file))
#     classifier = torch.load(path + "classifier_%s.pth" % (file))
#     model = classifier(encode)
#     return model
from sklearn.preprocessing import label_binarize
test_dataloader = data_loader(target_file)

def Caculate(weight):
    Encode=[]
    Classifier=[]
    for file in test_list:

        Ez, Cz=get_model(file)
        # Ez = Ez.to(device)
        # Cz = Cz.to(device)
        # model_temp.to(device)

        Encode.append(Ez)
        Classifier.append(Cz)
    acc = 0
    auc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = (labels.long()).to(device)
        Y = torch.zeros(batch_size, 3).to(device)
        for i in range(len(Encode)):
            Encode[i].eval()
            Classifier[i].eval()
            Y+=Classifier[i](Encode[i](data))*weight[i]
        acc = acc + (torch.max(Y, 1)[1] == labels).float().mean().item()
        y_one_hot = labels.cpu().detach().numpy()
        y_one_hot = label_binarize(y_one_hot, classes=np.arange(3))
        a = Y.cpu().detach().numpy()
        auc += metrics.roc_auc_score(y_one_hot, a, average='micro')
        # auc += metrics.roc_auc_score(labels.cpu(), torch.max(Y, 1)[1].cpu())
    accuracy = round(acc / float(len(test_dataloader)), 3)
    auc_temp = round(auc / float(len(test_dataloader)), 3)
    print("accuracy: %.3f " % (accuracy))
    return accuracy, auc_temp
print("len(test_list)", len(test_list))

weight=[1/len(test_list)]*len(test_list)

save_accuracy, save_auc = Caculate(weight)
save_dir = path1 = './seed/' + target_file[:-4] + '/result.xlsx'
import xlsxwriter
workbook = xlsxwriter.Workbook(save_dir)
worksheet = workbook.add_worksheet()
aa = final.__len__()
for i in range(int(aa)):
    worksheet.write(i, 0, name[i])  # 第i行0列
    worksheet.write(i, 1, final[i])  # 第i行1列
    worksheet.write(i, 8, auc_all[i])  # 第i行1列
bb = or_all.__len__()
for j in range(bb):
    worksheet.write(j, 5, or_all[j] )  # 第i行0列
    # worksheet.write(i, 1, final[i])  # 第i行1列

worksheet.write(0, 2, "max")
worksheet.write(1, 2, max_value)
worksheet.write(0, 3, "mean")
worksheet.write(1, 3, save_accuracy)
worksheet.write(0, 4, "k")
worksheet.write(1, 4, final_k)
worksheet.write(0, 6, test_list[0])
# worksheet.write(1, 6, test_list[1])
# worksheet.write(2, 6, test_list[2])
# worksheet.write(3, 6, test_list[3])
# worksheet.write(4, 6, test_list[4])

worksheet.write(0, 7, "or_data")
worksheet.write(1, 7, temp)

worksheet.write(0, 9, "mean_auc")
worksheet.write(1, 9, save_auc)


workbook.close()
print("over")
   