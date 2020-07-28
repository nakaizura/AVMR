import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)


class Disciminator(nn.Module):
    def __init__(self):
        super(Disciminator, self).__init__()
        self.semantic_size = 1024 # the size of visual and semantic comparison size
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 2048

        self.pack=128
        self.v2s_lt = nn.Linear(self.visual_feature_dim, self.semantic_size)
        self.lstm = nn.LSTM(self.pack,self.pack)
        self.l_lt = nn.Linear(self.pack, 1)
        self.s2s_lt = nn.Linear(self.sentence_embedding_size, self.semantic_size)

        self.fc1 = torch.nn.Conv2d(4096, 1000, kernel_size=1, stride=1)
        self.fc2 = torch.nn.Conv2d(1000, 100, kernel_size=1, stride=1)
        self.fc3 = torch.nn.Conv2d(100, 1, kernel_size=1, stride=1)
        # Initializing weights
        self.apply(weights_init)

    def cross_modal_comb(self, visual_feat, sentence_embed):
        batch_size = visual_feat.size(0)
        vv_feature = visual_feat.expand([batch_size,batch_size,self.semantic_size])
        ss_feature = sentence_embed.repeat(1,1,batch_size).view(batch_size,batch_size,self.semantic_size)

        concat_feature = torch.cat([vv_feature, ss_feature], 2)

        mul_feature = vv_feature * ss_feature
        add_feature = vv_feature + ss_feature

        comb_feature = torch.cat([mul_feature, add_feature, concat_feature], 2)

        return comb_feature


    def forward(self, visual_feature_train, sentence_embed_train):
        visual_feature_train = torch.transpose(visual_feature_train, 1, 2)
        #pack lstm
        visual_feature_train=torch.nn.utils.rnn.pack_padded_sequence(visual_feature_train,self.pack)
        
        visual_feature_train = torch.transpose(visual_feature_train, 1, 2)
        out, (hn,cn)= self.lstm(visual_feature_train)
        transformed_clip_train = self.l_lt(hn)
        transformed_clip_train = torch.transpose(transformed_clip_train, 1, 2)
        transformed_clip_train = self.v2s_lt(transformed_clip_train)
        transformed_clip_train_norm = F.normalize(transformed_clip_train, p=2, dim=1)

        
        transformed_sentence_train = self.s2s_lt(sentence_embed_train)
        transformed_sentence_train_norm = F.normalize(transformed_sentence_train, p=2, dim=1)
        

        cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm, transformed_sentence_train_norm)

        cross_modal_vec_train = cross_modal_vec_train.unsqueeze(0).permute(0, 3, 1, 2)
        mid_output = self.fc1(cross_modal_vec_train)
        mid_output = F.relu(mid_output)
        mid_output = self.fc2(mid_output)
        mid_output = F.relu(mid_output)
        mid_output = self.fc3(mid_output).squeeze(0)
        sim_score = F.sigmoid(mid_output)

        consistency=(transformed_clip_train_norm - transformed_sentence_train_norm).pow(2).squeeze().mean()

        return sim_score.squeeze(),consistency


