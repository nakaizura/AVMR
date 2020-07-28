import numpy as np
import os
import glob
import torch as th
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

def dataloader(path):
    dataloader=np.load(path)
    data_all=list()
    for i in range(len(dataloader)):
        data = list()
        data.append(dataloader[i]['video_info'])
        data.append(dataloader[i]['w_vec'])
        data.append(dataloader[i]['v_feature'])
        data.append(dataloader[i]['w_start'])
        data.append(dataloader[i]['w_end'])
        data.append(dataloader[i]['fps'])

        data_all.append(data)

    return data_all

def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    img = Image.open(img_path)
    img = transform(img)

    x = Variable(th.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()

    return y.tolist()

def resneti(path,start,end):
    files_list=list()
    for i in range(start,end):
        file_glob = os.path.join(path, str(i) + '.jpg')
        files_list.extend(glob.glob(file_glob))

    resnet50_feature_extractor = models.resnet50(pretrained=False)
    resnet50_feature_extractor.load_state_dict(th.load('/resnet50-19c8e357.pth'))
    resnet50_feature_extractor.fc = th.nn.Linear(2048, 2048)
    th.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = th.cuda.is_available()

    video_fearture=list()
    for x_path in [files_list[j] for j in range(len(files_list))]:
        sigle_feature = extractor(x_path, resnet50_feature_extractor, use_gpu)
        try:
            sigle_feature=extractor(x_path, resnet50_feature_extractor, use_gpu)
        except: break
        video_fearture.append(sigle_feature)
    feature=np.squeeze(np.array(video_fearture).mean(axis=0))

    return feature

def resnet(path,start,end):
    files_list=list()
    for i in range(start,end):
        file_glob = os.path.join(path, str(i) + '.jpg')
        files_list.extend(glob.glob(file_glob))

    resnet50_feature_extractor = models.resnet50(pretrained=False)
    resnet50_feature_extractor.load_state_dict(th.load('/resnet50-19c8e357.pth'))
    resnet50_feature_extractor.fc = th.nn.Linear(2048, 2048)
    th.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = th.cuda.is_available()

    video_fearture=list()
    for x_path in [files_list[j] for j in range(len(files_list))]:
        sigle_feature = extractor(x_path, resnet50_feature_extractor, use_gpu)
        try:
            sigle_feature=extractor(x_path, resnet50_feature_extractor, use_gpu)
        except: break
        video_fearture.append(sigle_feature)

    feature=np.squeeze(video_fearture)
    return feature

def get_bpr(T,F):
    r=0
    if T>F: r=0.1
    elif T==F:r=0.5
    else: r=1
    return r

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(abs(inter[1]-inter[0]))/(abs(union[1]-union[0]))
    return iou

def calculate_reward(Previou_IoU, current_IoU, t):
    if current_IoU > Previou_IoU and Previou_IoU>=0:
        reward = 1-0.00*t
    elif current_IoU <= Previou_IoU and current_IoU>=0:
        reward = -0.00*t
    else:
        reward = -1-0.00*t
    return reward

def compute_IoU_recall_top_n(topn, IoU, iuo_record):
    yes=0
    for i in range(len(iuo_record)):
        if max(iuo_record[i][:topn])>=IoU:
            yes=yes+1
    acc=yes/len(iuo_record)

    return acc
