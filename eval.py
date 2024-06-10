
import numpy as np

from mahjong_env.core import Mahjong
from mahjong_env.consts import ActionType
from mahjong_env.player_data import Action
from mahjong_env.base_bot import *
from mahjong_env.utils import *

from argparse import ArgumentParser
ACTION = {'PASS': 0, 'DRAW': 1, 'PLAY': 2,
          'CHOW': 3, 'PUNG': 4, 'KONG': 5, 'AnGang': 5, 'MELD_KONG': 6, 'Hu': 7, }
TILE_SET = {
    'W':0, 'T': 9, 'B': 18, 'F': 27, 'J':31
}


cnt=[0 for i in range(4)]
loss=[0 for i in range(4)]
huang=[0]

def random_test(east, south, west, north, args):
    
    env = Mahjong()
    res = env.init()
    # p1=[[0 for i in range(34)] for j in range(5)]
    p2=[[0 for i in range(34)] for j in range(16)]
    for player in range(4):
        for card in env.player_obs(player)['tiles']:
            p2[player][getID(card)]+=1
    # For p1
    #   0-3 for in hand
    #   4 for discard    
    # For p2
    #   0-3 for in hand
    #   4-7 for claiming
    #   8-11 for current discard
    #   12-15 for discard in history
    print(res)
    print('-------------')

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
            p2[player][getID(card)]+=1

            clearAllCurrent(p2)
        elif act=='PLAY':
            p2[player+8][getID(card)]+=1
            p2[player][getID(card)]-=1


            
        elif act=='PUNG':
            p2[player][getID(preCard)]-=2
            p2[player+4][getID(preCard)]+=3
            clearAllCurrent(p2)
            p2[player+8][getID(card)]+=1
            p2[player][getID(card)]-=1

            
        elif act=='CHOW':
            p2[player][getID(res[2])-1]-=1
            p2[player][getID(res[2])]-=1
            p2[player][getID(res[2])+1]-=1
            p2[player][getID(preCard)]+=1

            p2[player+4][getID(res[2])-1]+=1
            p2[player+4][getID(res[2])]+=1
            p2[player+4][getID(res[2])+1]+=1
            p2[player+4][getID(preCard)]-=1
            clearAllCurrent(p2)
            p2[player+8][getID(res[3])]+=1
            p2[player][getID(res[3])]-=1

        elif act=='KONG':
            p2[player][getID(card)]=0
            p2[player+4][getID(card)]=4
            clearAllCurrent(p2)


        elif act=='MELD_KONG':
            p2[player][getID(card)]-=1
            p2[player+4][getID(card)]+=1
            clearAllCurrent(p2)
            
        """
        for i in range(4):
            for j in range(34):
                for t in range(p[i][j]):
                    print(a.trans(j), end=' ')
            print()
        """
        obs = [env.player_obs(i) for i in range(0,4)]
        actions = []
        actions.append(east.action(p=p2, obs=obs[0]))
        actions.append(south.action(p=p2, obs=obs[1]))
        actions.append(west.action(p=p2, obs=obs[2]))
        actions.append(north.action(p=p2, obs=obs[3]))
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
    parser=ArgumentParser()
    parser.add_argument('--east', default=0, type=int) 
    parser.add_argument('--south', default=0, type=int)
    parser.add_argument('--west', default=0, type=int)
    parser.add_argument('--north', default=0, type=int)
    parser.add_argument('--round', default=100, type=int)
    args = parser.parse_args()

    # NN1Discard
    NN1DiscardPath='./model/0/discard.pth'

    # NN23Discard
    NN23DiscardPath='./model/1/discard.pth'

    # NN1Claiming
    NN1chipath='./model/2/chi.pth'
    NN1pengpath='./model/2/peng.pth'
    NN1gangpath='./model/2/gang.pth'
    NN1bugangpath='./model/2/bugang.pth'
    NN1angangpath='./model/2/angang.pth'
    
    # NN2Claiming
    NN2chipath='./model/3/Chi.pth'
    NN2pengpath='./model/3/Peng.pth'
    NN2gangpath='./model/3/Gang.pth'
    NN2bugangpath='./model/3/Bugang.pth'
    NN2angangpath='./model/3/Angang.pth'
    
    # NN3Claiming
    NN3ClaimingPath='./model/4/claiming.pth'
    
    if args.east==0:
        east=RandomMahjongBot()
    elif args.east==1:
        east=AstarMahjongBot()
    elif args.east==2:
        east=ExpectiMaxMahjongBot()
    elif args.east==3:
        east=NN1agent()
        east.init(NN1DiscardPath, NN1chipath, NN1pengpath, NN1gangpath, NN1bugangpath, NN1angangpath)
    elif args.east==4:
        east=NN2agent()
        east.init(NN23DiscardPath, NN2chipath, NN2pengpath, NN2gangpath, NN2bugangpath, NN2angangpath)
    elif args.east==5:
        east=NN3agent()
        east.init(NN23DiscardPath, NN3ClaimingPath)
    
    if args.south==0:
        south=RandomMahjongBot()
    elif args.south==1:
        south=AstarMahjongBot()
    elif args.south==2:
        south=ExpectiMaxMahjongBot()
    elif args.south==3:
        south=NN1agent()
        south.init(NN1DiscardPath, NN1chipath, NN1pengpath, NN1gangpath, NN1bugangpath, NN1angangpath)
    elif args.south==4:
        south=NN2agent()
        south.init(NN23DiscardPath, NN2chipath, NN2pengpath, NN2gangpath, NN2bugangpath, NN2angangpath)
    elif args.south==5:
        south=NN3agent()
        south.init(NN23DiscardPath, NN3ClaimingPath)
    
    if args.west==0:
        west=RandomMahjongBot()
    elif args.west==1:
        west=AstarMahjongBot()
    elif args.west==2:
        west=ExpectiMaxMahjongBot()
    elif args.west==3:
        west=NN1agent()
        west.init(NN1DiscardPath, NN1chipath, NN1pengpath, NN1gangpath, NN1bugangpath, NN1angangpath)
    elif args.west==4:
        west=NN2agent()
        west.init(NN23DiscardPath, NN2chipath, NN2pengpath, NN2gangpath, NN2bugangpath, NN2angangpath)
    elif args.west==5:
        west=NN3agent()
        west.init(NN23DiscardPath, NN3ClaimingPath)
    
    if args.north==0:
        north=RandomMahjongBot()
    elif args.north==1:
        north=AstarMahjongBot()
    elif args.north==2:
        north=ExpectiMaxMahjongBot()
    elif args.north==3:
        north=NN1agent()
        north.init(NN1DiscardPath, NN1chipath, NN1pengpath, NN1gangpath, NN1bugangpath, NN1angangpath)
    elif args.north==4:
        north=NN2agent()
        north.init(NN23DiscardPath, NN2chipath, NN2pengpath, NN2gangpath, NN2bugangpath, NN2angangpath)
    elif args.north==5:
        north=NN3agent()
        north.init(NN23DiscardPath, NN3ClaimingPath)
    
    for i in range(args.round):
        random_test(east, south, west, north, args)
    print('The # of winning games:')
    for i in range(4):
        print(f'player {i}: {cnt[i]}')
    print()
    print('The # of lose games:')
    for i in range(4):
        print(f'player {i}: {loss[i]}')
if __name__ == '__main__':
    main()
