import json
import numpy as np
from tqdm import tqdm 
from mahjong_env.core import Mahjong
from mahjong_env.consts import ActionType
from mahjong_env.player_data import Action
from mahjong_env.base_bot import RandomMahjongBot

ROUND_WIND = {'东': 0, '南': 1, '西': 2, '北': 3}
INIT_TILES_NUM = 13
ACTION = {'Pass': 0, 'Draw': 1, 'Play': 2,
          'Chi': 3, 'Peng': 4, 'Gang': 5, 'AnGang': 5, 'BuGang': 6, 'Hu': 7, }
TILE_SET = {
    'W':0, 'T': 9, 'B': 18, 'F': 27, 'J':31
}

NO_VALID_ACTION = ['补花']
NUM_PLAYERS = 4
peng=[]
chi=[]
labelChi=[]
gang=[]
angang=[]
bugang=[]
Pass=[]
discard=[]
labelDiscard=[]
def getID(tile):
    return TILE_SET[tile[0]]+int(tile[1])-1
def genOne(tile):
    ret=[0 for i in range(34)]
    ret[getID(tile)]=1
    return ret
def clearAllCurrent(p):
    for i in range(34):
        for j in range(4):
            p[12+j][i]+=p[8+j][i]
            p[8+j][i]=0

def play_round(lines, env):
    round_wind=int(lines[1][1])
    tile0=lines[2][3:16]
    tile1=lines[3][3:16]
    tile2=lines[4][3:16]
    tile3=lines[5][3:16]
    p=[[0 for i in range(34)] for j in range(16)]
    # 0-3 for in hand
    # 4-7 for claiming
    # 8-11 for current discard
    # 12-15 for discard in history
    for i in tile0:
        p[0][getID(i)]+=1
    for i in tile1:
        p[1][getID(i)]+=1
    for i in tile2:
        p[2][getID(i)]+=1
    for i in tile3:
        p[3][getID(i)]+=1
    
    firstPlayer=0
    pre=""
    preAct=''
    prePlayer=-1
    for i in range(6, len(lines)-3):
        _, player, act, tile, *__=lines[i]
        # print(_, player, act, tile)
        player=int(player)
        if act=='Draw': 
            if preAct=='Play':
                for er in range(1,4):
                   """ Pass.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12])    """
            
            p[player][getID(tile)]+=1
            clearAllCurrent(p)
        elif act=='Play':
            if len(discard)<1000000:
                er=0
                discard.append(p[(player+er)%4]+\
                                    p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                    p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3 )%4+8]+\
                                    p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12]) 
                tmp=[0 for i in range(34)]
                tmp[getID(tile)]=1
                labelDiscard.append(tmp)
            p[player+8][getID(tile)]+=1
            p[player][getID(tile)]-=1
        elif act=='Peng':
            er=0
            
            """peng.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12]) """   
            p[player][getID(pre)]-=2
            p[player+4][getID(pre)]+=3
            clearAllCurrent(p)
        elif act=='Chi':
            er=0
            chi.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12])
            tmp=[0 for i in range(34)]
            tmp[getID(tile)]=1
            labelChi.append(tmp)
            p[player][getID(tile)-1]-=1
            p[player][getID(tile)]-=1
            p[player][getID(tile)+1]-=1
            p[player][getID(pre)]+=1

            p[player+4][getID(tile)-1]+=1
            p[player+4][getID(tile)]+=1
            p[player+4][getID(tile)+1]+=1
            p[player+4][getID(pre)]-=1
            clearAllCurrent(p)
        elif act=='Gang':
            er=0
            """gang.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12])"""    
            p[player][getID(tile)]-=3
            p[player+4][getID(tile)]+=4
            clearAllCurrent(p)
        elif act=='AnGang':
            er=0
            """angang.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12])"""    
            p[player][getID(tile)]-=4
            clearAllCurrent(p)
        elif act=='BuGang':
            er=0
            """bugang.append(p[(player+er)%4]+\
                                p[(player+er)%4+4]+p[(player+er+1)%4+4]+p[(player+er+2)%4+4]+p[(player+er+3)%4+4]+\
                                p[(player+er)%4+8]+p[(player+er+1)%4+8]+p[(player+er+2)%4+8]+p[(player+er+3)%4+8]+\
                                p[(player+er)%4+12]+p[(player+er+1)%4+12]+p[(player+er+2)%4+12]+p[(player+er+3)%4+12])"""    
            p[player][getID(tile)]-=1
            p[player+4][getID(tile)]+=1
            clearAllCurrent(p)
        pre=tile
        preAct=act
        prePlayer=player


    

if __name__ == '__main__':
    file = open('mahjong_data/data/data.txt', 'r')
    # target=open('mahjong_data/data/dataset.txt', 'w')
    lines=[]
    env=Mahjong()
    # pbar=tqdm(file, total=len(file.readlines()))
    t=0
    for line in file:
        if line=='\n':
            play_round(lines, env)
            lines.clear()
            t+=1
            print(t)
            #if t>10000:
                #break
        else:
            lines.append(line.rstrip().split(' '))


    #np.save('dataset/peng',np.array(peng))
    #np.save('dataset/chi',np.array(chi))
    #np.save('dataset/labelchi',np.array(labelChi))
    np.save('dataset/discard',np.array(discard))
    np.save('dataset/labeldiscard',np.array(labelDiscard))
    #np.save('dataset/gang',np.array(gang))
    #np.save('dataset/bugang',np.array(bugang))
    #np.save('dataset/angang',np.array(angang))
    #np.save('dataset/Pass',np.array(Pass))
    #print(len(peng))
    #print(len(chi))
    print(len(discard))
    #print(len(bugang))
    #print(len(angang))    
    #print(len(Pass))
