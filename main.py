from MADDPG import MADDPG
from Disciminator import Disciminator
import numpy as np
import torch as th
import collections
import warnings
from utils import *
import heapq


def init_state(path,video_info,w_vec,globle_vec):
    location=th.tensor([0.25,0.75])
    path=path+video_info
    start=int(int(video_info.split('-')[2])*0.25)
    end=int(int(video_info.split('-')[2])*0.75)
    local_vec=th.tensor(resneti(path,start,end)).float()
    state = th.cat((w_vec, local_vec,globle_vec.float(),location), 0).unsqueeze(0).cuda()
    return state

def state_update(w_start,w_end,state,action,unit,step,fi):
    for n in range(n_agents):
        location=state[n][-2:]
        old=location.cpu().numpy()
        flag=np.argmax(action[n])

        old_iou = calculate_IoU(location, [w_start, w_end])
        if flag==0:
            old[0]=location[0]-unit
            old[1] = location[1] - unit
        elif flag==1:
            old[0] = location[0] + unit
            old[1] = location[1] + unit
        elif flag==2:
            old[0] = location[0] - unit
        elif flag==3:
            old[0] = location[0] + unit
        elif flag==4:
            old[1] = location[1] - unit
        elif flag==5:
            old[1] = location[1] + unit
        else:
            stop=True
            return state,[fi],old_iou

        if old[0]<0 or old[1]>1 or old[0]>=old[1]:
            stop=True
            return state,[fi],old_iou

        now_iou=calculate_IoU(old,[w_start,w_end])
        reward=[calculate_reward(old_iou,now_iou,step)]
        state[n][-2] = th.tensor(old[0]).cuda()
        state[n][-1] = th.tensor(old[1]).cuda()
        state_new = state

    return state_new,reward,now_iou


np.random.seed(1234)
th.manual_seed(1111)
n_agents = 1
n_states = 8898
n_actions = 6
capacity = 1000000
batch_size = 1

n_episode = 10
episodes_before_train = 1

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                    episodes_before_train)
distor= Disciminator().cuda()
optimizer = th.optim.Adam(distor.parameters(), lr=0.001)

def train():
    print('###############TRAIN##################')
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

    rr = np.zeros((n_agents,))
    loader=dataloader('/feature_all_train.npy')
    cut_path='/WorkplaceTrain/'
    dataload = th.utils.data.DataLoader(dataset=loader,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    for batch_idx, all_f in enumerate(dataload):
        if(batch_idx>100):break
        w_start = float(all_f[:][3][0])
        w_vec = all_f[:][1][0]
        w_end = float(all_f[:][4][0])
        v_vec = all_f[:][2][0]
        video_info = all_f[:][0][0]
        fps = all_f[:][5][0]

        print(batch_idx,video_info)
        start_f=(int(w_start*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])
        end_f=(int(w_end*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])


        # generator
        # True sample
        Truestart = int(int(video_info.split('-')[2]) * start_f)
        Trueend = int(int(video_info.split('-')[2]) * end_f)
        Truevec = th.tensor(resnet(cut_path + video_info, Truestart, Trueend)).float()
        T_score, T_con = distor(th.tensor(Truevec.unsqueeze(0)).cuda(), th.tensor(w_vec[0].unsqueeze(0)).cuda())
        # Flase sample
        quick = 0.05  # moving step
        fi = -1  # 1.abnormal penalty
        p1 = 0.8
        p2 = 0.2
        state = init_state(cut_path, video_info, th.squeeze(th.squeeze(w_vec, 0)), v_vec)
        for i_episode in range(n_episode):
            #select action
            action = maddpg.select_action(state).data.cpu()
            state_, iou_reward, iou_now = state_update(start_f,end_f,state,action.numpy(),quick,i_episode,fi)

            #get reward
            localstart=int(int(video_info.split('-')[2])*state_[0][-2])
            localend=int(int(video_info.split('-')[2])*state_[0][-1])
            localvec = th.tensor(resnet(cut_path+video_info, localstart, localend)).float()
            score,con=distor(th.tensor(localvec.unsqueeze(0)).cuda(), th.tensor(w_vec[0].unsqueeze(0)).cuda())

            bpr_reward=get_bpr(T_score,score)
            reward = p1*iou_reward[0] * score - bpr_reward*th.log(th.sigmoid(T_score - score)) - p2 * T_con-0.01021*i_episode

            maddpg.memory.push(state.data, action, state_, th.tensor([reward]))
            state = state_

            c_loss, a_loss = maddpg.update_policy()
            #print(c_loss, a_loss)

            maddpg.episode_done += 1
            print('Episode: %d, reward = %f,IOU:%f' % (i_episode, reward,iou_now))



        #disciminator
        state = init_state(cut_path, video_info, th.squeeze(th.squeeze(w_vec, 0)), v_vec)
        reward=0
        for i_episode in range(n_episode):
            #select action
            action = maddpg.select_action(state).data.cpu()
            state_, iou_reward, iou_now = state_update(start_f,end_f,state,action.numpy(),quick,i_episode,fi)

            #get reward
            localstart=int(int(video_info.split('-')[2])*state_[0][-2])
            localend=int(int(video_info.split('-')[2])*state_[0][-1])
            localvec = th.tensor(resnet(cut_path+video_info, localstart, localend)).float()
            score,con=distor(th.tensor(localvec.unsqueeze(0)).cuda(), th.tensor(w_vec[0].unsqueeze(0)).cuda())
            bpr_reward = get_bpr(T_score, score)

            reward += p1*iou_reward[0] * score - bpr_reward*th.log(th.sigmoid(T_score - score))
            state = state_

        optimizer.zero_grad()
        reward.backward()
        optimizer.step()
        print('loss = %f,rec_IOU:%f' % (reward,iou_now))
        print('over--------------------')


def test():
    print('###############TEST##################')
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

    rr = np.zeros((n_agents,))
    loader = dataloader('/feature_all_test.npy')
    cut_path = '/WorkplaceTest/'
    dataload = th.utils.data.DataLoader(dataset=loader,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4)

    iou_record=[]
    for batch_idx, all_f in enumerate(dataload):
        if (batch_idx > 500): break
        w_start = float(all_f[:][3][0])
        w_vec = all_f[:][1][0]
        w_end = float(all_f[:][4][0])
        v_vec = all_f[:][2][0]
        video_info = all_f[:][0][0]
        fps = all_f[:][5][0]

        print(batch_idx, video_info)
        start_f = (int(w_start * fps.data) - int(video_info.split('-')[1])) / int(video_info.split('-')[2])
        end_f = (int(w_end * fps.data) - int(video_info.split('-')[1])) / int(video_info.split('-')[2])


        quick = 0.05  # moving step
        fi = -1  # 1.abnormal penalty

        state = init_state(cut_path, video_info, th.squeeze(th.squeeze(w_vec, 0)), v_vec)
        score_all=[]
        iou_all=[]
        for i_episode in range(n_episode):
            # generator
            # select action
            action = maddpg.select_action(state).data.cpu()
            state_, iou_reward, iou_now = state_update(start_f, end_f, state, action.numpy(), quick, i_episode, fi)

            # disciminator
            localstart = int(int(video_info.split('-')[2]) * state_[0][-2])
            localend = int(int(video_info.split('-')[2]) * state_[0][-1])
            localvec = th.tensor(resnet(cut_path + video_info, localstart, localend)).float()
            score, con = distor(th.tensor(localvec.unsqueeze(0)).cuda(), th.tensor(w_vec.unsqueeze(0)).cuda())
            score_all.append(score)
            iou_all.append(iou_now)
            state = state_

        rec=map(score_all.index,heapq.nlargest(5,score_all))
        R_rec=[iou_all[i] for i in rec]
        #rec=iou_all[score_all.index(max(score_all))]

        print(batch_idx,R_rec,'over--------------------')
        iou_record.append(R_rec)

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    R=[1,3,5]

    for r in R:
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            acc = compute_IoU_recall_top_n(r, IoU, iou_record)
            print('R@%d IOU=%f : %f'%(r,IoU,acc))
        print()



if __name__ == '__main__':
    train_epoch = 100
    test_epoch = 2

    train()
    test()
