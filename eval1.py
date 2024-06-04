import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from mahjong_env.core import Mahjong
from mahjong_env.consts import ActionType
from mahjong_env.player_data import Action
from mahjong_env.base_bot import RandomMahjongBot, AstarMahjongBot, BaseMahjongBot, RLMahjongBot

from RL.model3 import ClaimingModel
from RL.discard2 import DiscardModel as D

ACTION = {'PASS': 0, 'DRAW': 1, 'PLAY': 2,
          'CHOW': 3, 'PUNG': 4, 'KONG': 5, 'AnGang': 5, 'MELD_KONG': 6, 'Hu': 7, }
TILE_SET = {
    'W':0, 'T': 9, 'B': 18, 'F': 27, 'J':31
}

class SLagent1(BaseMahjongBot):
    def init(self, discardModelPath, chiModelPath, pengModelPath, gangModelPath, bugangModelPath, angangModelPath):

        self.pengModel=ClaimingModel()
        self.pengModel.load_state_dict(torch.load(pengModelPath))
        self.gangModel=ClaimingModel()
        self.gangModel.load_state_dict(torch.load(gangModelPath))
        self.bugangModel=ClaimingModel()
        self.bugangModel.load_state_dict(torch.load(bugangModelPath))
        self.angangModel=ClaimingModel()
        self.angangModel.load_state_dict(torch.load(angangModelPath))
        self.chiModel=ClaimingModel()
        self.chiModel.load_state_dict(torch.load(chiModelPath))
        self.discardModel=D()
        self.discardModel.load_state_dict(torch.load(discardModelPath))

    def chi(self, input):
        input=torch.tensor(input).float()
        needChi=self.chiModel(input)
        if torch.argmax(needChi)==1:
            return 1
        return 0    

    def peng(self, input):
        input=torch.tensor(input).float()
        if torch.argmax(self.pengModel(input)).item()==1:
            return 1
        return 0

    def gang(self, input):
        input=torch.tensor(input).float()
        if torch.argmax(self.gangModel(input)).item()==1:
            return 1
        return 0

    def bugang(self, input):
        input=torch.tensor(input).float()
        if torch.argmax(self.bugangModel(input)).item()==1:
            return 1
        return 0

    def angang(self, input):
        input=torch.tensor(input).float()
        if torch.argmax(self.angangModel(input)).item()==1:
            return 1
        return 0
    def trans(self, card):
        if card<9:
            return 'W'+str(card+1)
        if card<18:
            return 'T'+str(card-8)
        if card<27:
            return 'B'+str(card-17)
        if card<31:
            return 'F'+str(card-26)
        return 'J'+str(card-30)
    def discard(self, input):
        input=torch.tensor(input).float()
        card=torch.topk(self.discardModel(input), 34).indices
        #print(card)
        #print(input[0:34])
        for i in range(34):
            if input[card[i].item()]>0:
                return self.trans(card[i].item())
        
    def selectAction(self, p, obs: dict):
        pn=[0 for i in range(34)]
        dc=[0 for i in range(34)]
        for player in range(4):
            for i in range(34):
                pn[i]+=p[player+4][i]
                dc[i]+=p[player+8][i]
                pn[i]+=p[player+12][i]
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(0, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            if last_player != player:
                return pass_action
            else:
                if self.check_hu(obs)>=8:
                    return Action(player, ActionType.HU, None)
                
                if self.check_kong(obs):
                    
                    ret=self.angang(p[player]+pn+dc)
                    
                    if ret:
                        return Action(player, ActionType.KONG, obs['last_tile'])
                if self.check_meld_kong(obs):
                    ret=self.bugang(p[player]+pn+dc)
                    if ret:
                        return Action(player, ActionType.MELD_KONG, obs['last_tile'])

                discard=self.discard(p[player]+pn)
                # assert(p[0][getID(discard)]>0)
                return Action(player, ActionType.PLAY, discard)
            

        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        if last_player == player:
            return pass_action
        
        if self.check_hu(obs)>=8:
            return Action(player, ActionType.HU, None)
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
        if self.check_kong(obs):
            ret=self.gang(p[player]+pn+dc)
            if ret:
                return Action(player, ActionType.KONG, None)
        if self.check_meld_kong(obs):
            ret=self.bugang(p[player]+pn+dc)
            if ret:
                return Action(player, ActionType.MELD_KONG, None)
        if self.check_pung(obs):
            ret=self.peng(p[player]+pn+dc)
            if ret==1:
                tiles = obs['tiles'].copy()
                idx=getID(obs['last_tile'])
                p[player][idx]-=2
                p[player+4][idx]+=3
                play_tile = self.discard(p[player]+pn)
                # assert(p[0][getID(play_tile)]>0)
                p[player][idx]+=2
                p[player+4][idx]-=3
                return Action(player, ActionType.PUNG, play_tile)
        chow_list = self.check_chow(obs)
        if len(chow_list) != 0:
            ret=self.chi(p[player]+pn+dc)
            if ret==1:
                tmp=getID(chow_list[0])
                for i in range(tmp - 1, tmp + 2):
                    if i == getID(obs['last_tile']):
                        continue
                    else:
                        #print(f'{ret[i]}{i}')
                        #tiles.remove(f'{chow_t}{i}')
                        p[player][i]-=1
                    p[player+4][i]+=1
                discard=self.discard(p[player]+pn)
                # assert(p[0][getID(discard)]>0)
                for i in range(tmp - 1, tmp + 2):
                    if i == getID(obs['last_tile']):
                        continue
                    else:
                        #print(f'{ret[i]}{i}')
                        #tiles.remove(f'{chow_t}{i}')
                        p[player][i]+=1
                    p[player+4][i]-=1
                return Action(player, ActionType.CHOW, f'{self.trans(tmp)} {discard}')
        return pass_action

def getID(tile):
    return TILE_SET[tile[0]]+int(tile[1])-1
def clearAllCurrent(p):
    for i in range(34):
        for j in range(4):
            p[12+j][i]+=p[8+j][i]
            p[8+j][i]=0
cnt=[0 for i in range(4)]
loss=[0 for i in range(4)]

def random_test(a):
    
    env = Mahjong(random_seed=time.time())
    res = env.init()
    p=[[0 for i in range(34)] for j in range(16)]
    for player in range(4):
        for card in env.player_obs(player)['tiles']:
            p[player][getID(card)]+=1
    # 0-3 for in hand
    # 4-7 for claiming
    # 8-11 for current discard
    # 12-15 for discard in history
    print(res)
    print('-------------')
    agent = RandomMahjongBot()
    Astar = AstarMahjongBot()
    RL = RLMahjongBot()
    preCard=''
    while not env.done:
        print(env.observation())
        res=str(res).split(' ')
        player, act=int(res[0]), res[1]
        if act in {'DRAW', 'PLAY', 'PUNG', 'MELD_KONG', 'CHOW'}:
            card=res[2]
        elif act=='KONG':
            card=preCard
        # print()
        if act=='DRAW': 
            p[player][getID(card)]+=1
            clearAllCurrent(p)
        elif act=='PLAY':
            p[player+8][getID(card)]+=1
            p[player][getID(card)]-=1
        elif act=='PUNG':
            p[player][getID(preCard)]-=2
            p[player+4][getID(preCard)]+=3
            clearAllCurrent(p)
            p[player+8][getID(card)]+=1
            p[player][getID(card)]-=1

        elif act=='CHOW':
            p[player][getID(res[2])-1]-=1
            p[player][getID(res[2])]-=1
            p[player][getID(res[2])+1]-=1
            p[player][getID(preCard)]+=1

            p[player+4][getID(res[2])-1]+=1
            p[player+4][getID(res[2])]+=1
            p[player+4][getID(res[2])+1]+=1
            p[player+4][getID(preCard)]-=1
            clearAllCurrent(p)
            p[player+8][getID(res[3])]+=1
            p[player][getID(res[3])]-=1
        elif act=='KONG':
            p[player][getID(card)]=0
            p[player+4][getID(card)]=4
            clearAllCurrent(p)
        elif act=='MELD_KONG':
            p[player][getID(card)]-=1
            p[player+4][getID(card)]+=1
            clearAllCurrent(p)
        """
        for i in range(4):
            for j in range(34):
                for t in range(p[i][j]):
                    print(a.trans(j), end=' ')
            print()
        """
        obs = [env.player_obs(i) for i in range(0,4)]
        actions = []
        tmp=p.copy()
        actions.append(a.selectAction(tmp,env.player_obs(0)))
        actions.append(agent.action(obs[1]))
        actions.append(agent.action(obs[2]))
        actions.append(agent.action(obs[3]))
        #[agent.action(ob) for ob in obs]
        preCard=res[-1]
        #print(actions)
        gun=env.player_obs(0)['last_player']
        res = env.step(actions)
        if (str(res)[2]=='H'):
            cnt[int(str(res)[0])]+=1
        

        print(res, actions)
    
   # print(gun)
    if str(res)[2]=='H' and gun!=int(str(res)[0]):
        loss[gun]+=1
        
    if env.fans is not None:
        print('Fans:', env.fans)
    print('Rewards:', env.rewards)
    

def main():
    chipath='./RL/model/1/chi.pth'
    pengpath='./RL/model/1/peng.pth'
    gangpath='./RL/model/1/gang.pth'
    bugangpath='./RL/model/1/bugang.pth'
    angangpath='./RL/model/1/angang.pth'
    discardpath='./RL/model/discard.pth'
    
    a=SLagent1()
    a.init(discardpath, chipath, pengpath, gangpath, bugangpath, angangpath)
    for i in range(500):
        random_test(a)
    print(cnt)
    print(loss)
if __name__ == '__main__':
    main()
