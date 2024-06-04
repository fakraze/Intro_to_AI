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

Play=[]
label=[]
def getID(tile):
    return TILE_SET[tile[0]]+int(tile[1])-1
def genOne(tile):
    ret=[0 for i in range(34)]
    ret[getID(tile)]=1
    return ret
def play_round(lines, env):
    round_wind=int(lines[1][1])
    tile0=lines[2][3:16]
    tile1=lines[3][3:16]
    tile2=lines[4][3:16]
    tile3=lines[5][3:16]
    p=[[0 for i in range(34)] for j in range(5)] # 4 for shown
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
    for i in range(6, len(lines)-3):
        _, player, act, tile, *__=lines[i]
        # print(_, player, act, tile)
        player=int(player)
        if act=='Draw':
            p[player][getID(tile)]+=1
        elif act=='Play':
            Play.append(p[player]+p[4])
            tmp=[0 for i in range(34)]
            tmp[getID(tile)]=1
            label.append(tmp)
            p[player][getID(tile)]-=1
            p[4][getID(tile)]+=1
        elif act=='Peng':
            p[player][getID(tile)]-=2
            p[4][getID(tile)]+=2

        elif act=='Chi':
            p[player][getID(tile)-1]-=1
            p[player][getID(tile)]-=1
            p[player][getID(tile)+1]-=1
            p[player][getID(pre)]+=1

            p[4][getID(tile)-1]+=1
            p[4][getID(tile)]+=1
            p[4][getID(tile)+1]+=1
            p[4][getID(pre)]-=1
            
        elif act=='Gang':
            p[player][getID(tile)]-=3
            p[4][getID(tile)]+=3
        elif act=='AnGang':
            p[player][getID(tile)]-=4
        elif act=='BuGang':
            p[player][getID(tile)]-=1
            p[4][getID(tile)]+=1
            
        pre=tile

    

    

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
            if t>20000: 
                break
        else:
            lines.append(line.rstrip().split(' '))


    np.save('dataset/play',np.array(Play))
    np.save('dataset/label', np.array(label))
    print(len(Play))
