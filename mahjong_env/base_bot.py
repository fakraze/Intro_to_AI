
from .ten import *

import random
from collections import Counter
from typing import List
from .consts import ActionType, ClaimingType
from .player_data import Action
from MahjongGB import MahjongFanCalculator
from .ten import *
import torch
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BaseMahjongBot:
    @staticmethod
    def check_hu(obs) -> int:
        player = obs['player_id']
        win_tile = obs['last_tile']
        claimings = []
        for claiming in obs['claimings']:
            if claiming.claiming_type == ClaimingType.CHOW:
                data = claiming.data
            else:
                data = (claiming.data - player + 4) % 4
            claimings.append((claiming.claiming_type, claiming.tile, data))
        claimings = tuple(claimings)
        tiles = tuple(obs['tiles'])

        flower_count = 0  # 补花数
        is_self_draw = obs['last_operation'] == ActionType.DRAW  # 自摸
        is_fourth_tile = obs['played_tiles'][win_tile] == 4  # 绝张
        is_kong = obs['last_operation'] == ActionType.MELD_KONG  # 杠
        is_kong |= len(obs['request_hist']) >= 2 \
                   and obs['request_hist'][-2].act_type == ActionType.KONG \
                   and obs['request_hist'][-2].player == player
        is_last_tile = obs['tile_count'][(player + 1) % 4] == 0  # 牌墙最后一张
        player_wind = player  # 门风
        round_wind = obs['round_wind']  # 圈风

        try:
            result = MahjongFanCalculator(claimings, tiles, win_tile, flower_count, is_self_draw, is_fourth_tile,
                                          is_kong, is_last_tile, player_wind, round_wind)
            return sum([res[0] for res in result])
        except Exception as exception:
            if str(exception) == "ERROR_NOT_WIN":
                return -1
            raise

    @staticmethod
    def check_kong(obs) -> bool:
        player = obs['player_id']
        if obs['tile_count'][player] == 0 or obs['tile_count'][(player + 1) % 4] == 0:
            return False
        if obs['tiles'].count(obs['last_tile']) == 3:
            return True
        return False

    @staticmethod
    def check_meld_kong(obs) -> bool:
        player = obs['player_id']
        if obs['tile_count'][player] == 0 or obs['tile_count'][(player + 1) % 4] == 0:
            return False
        for claiming in obs['claimings']:
            if claiming.claiming_type == ClaimingType.PUNG and claiming.tile == obs['last_tile']:
                return True
        return False

    @staticmethod
    def check_pung(obs) -> bool:
        if obs['tiles'].count(obs['last_tile']) == 2:
            return True
        return False

    @staticmethod
    def check_chow(obs) -> List[str]:
        if (obs['last_player'] - obs['player_id']) % 4 != 3:
            return []
        tile_t, tile_v = obs['last_tile']
        tile_v = int(tile_v)
        if tile_t in 'FJ':
            return []

        tiles = obs['tiles']
        chow_list = []
        if tile_v >= 3:
            if tiles.count(f'{tile_t}{tile_v - 1}') and tiles.count(f'{tile_t}{tile_v - 2}'):
                chow_list.append(f'{tile_t}{tile_v - 1}')
        if 2 <= tile_v <= 8:
            if tiles.count(f'{tile_t}{tile_v - 1}') and tiles.count(f'{tile_t}{tile_v + 1}'):
                chow_list.append(f'{tile_t}{tile_v}')
        if tile_v <= 7:
            if tiles.count(f'{tile_t}{tile_v + 1}') and tiles.count(f'{tile_t}{tile_v + 2}'):
                chow_list.append(f'{tile_t}{tile_v + 1}')
        return chow_list

    def action(self, obs: dict) -> Action:
        raise NotImplementedError


class RandomMahjongBot(BaseMahjongBot):
    @staticmethod
    def choose_play(tiles):
        cnt = Counter(tiles)
        single = [c for c, n in cnt.items() if n == 1]
        if len(single) == 0:
            double = [c for c, n in cnt.items() if n == 2]
            return random.choice(double)
        winds = [c for c in single if c[0] in 'FJ']
        if len(winds) != 0:
            return random.choice(winds)
        return random.choice(single)

    def action(self, obs: dict) -> Action:
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(player, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            if last_player != player:
                return pass_action
            else:
                fan = self.check_hu(obs)
                if fan >= 8:
                    return Action(player, ActionType.HU, None)

                if self.check_kong(obs):
                    return Action(player, ActionType.KONG, obs['last_tile'])
                if self.check_meld_kong(obs):
                    return Action(player, ActionType.MELD_KONG, obs['last_tile'])
                play_tile = self.choose_play(obs['tiles'] + [obs['last_tile']])
                return Action(player, ActionType.PLAY, play_tile)

        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        if last_player == player:
            return pass_action

        fan = self.check_hu(obs)
        if fan >= 8:
            return Action(player, ActionType.HU, None)
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
        if self.check_kong(obs):
            return Action(player, ActionType.KONG, None)
        if self.check_pung(obs):
            tiles = obs['tiles'].copy()
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            play_tile = self.choose_play(tiles)
            return Action(player, ActionType.PUNG, play_tile)

        chow_list = self.check_chow(obs)
        if len(chow_list) != 0:
            chow_tile = random.choice(chow_list)
            # print(chow_list)
            chow_t, chow_v = chow_tile[0], int(chow_tile[1])
            tiles = obs['tiles'].copy()
            for i in range(chow_v - 1, chow_v + 2):
                if i == int(obs['last_tile'][1]):
                    
                    continue
                else:
                    # print(f'{chow_t}{i}')
                    tiles.remove(f'{chow_t}{i}')
            play_tile = self.choose_play(tiles)
            return Action(player, ActionType.CHOW, f'{chow_tile} {play_tile}')
        return pass_action

class AstarMahjongBot(BaseMahjongBot):
    @staticmethod
    def choose_play(tiles, playedtitles):
        reward = 0
        playtile = None
        for tile in tiles:
            tiles2 = tiles.copy()
            tiles2.remove(tile)
            tilestuple = tuple(tiles2)
            currentReward = Shanten(tilestuple, playedtitles)
            if currentReward > reward:
                reward = currentReward
                playtile = tile
            # print("tile")
            # print(tile)
            # print("currentreward")
            # print(currentReward )
        return playtile
        

    def action(self, obs: dict) -> Action:
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(player, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            if last_player != player:
                return pass_action
            else:
                fan = self.check_hu(obs)
                if fan >= 1:
                    return Action(player, ActionType.HU, None)
                if self.check_kong(obs):
                    return Action(player, ActionType.KONG, obs['last_tile'])
                if self.check_meld_kong(obs):
                    return Action(player, ActionType.MELD_KONG, obs['last_tile'])
                # print("obs['tiles']")
                # print(obs['tiles'])
                # print("[obs['last_tile']]")
                # print([obs['last_tile']])
                play_tile = self.choose_play(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])
                return Action(player, ActionType.PLAY, play_tile)

        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        if last_player == player:
            return pass_action

        fan = self.check_hu(obs)
        if fan >= 1:
            
            return Action(player, ActionType.HU, None)
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
        if self.check_kong(obs):
            return Action(player, ActionType.KONG, None)
        if self.check_pung(obs):
            tiles = obs['tiles'].copy()
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            play_tile = self.choose_play(tiles, obs['played_tiles'])
            return Action(player, ActionType.PUNG, play_tile)

        chow_list = self.check_chow(obs)
        if len(chow_list) != 0:
            chow_tile = random.choice(chow_list)
            chow_t, chow_v = chow_tile[0], int(chow_tile[1])
            tiles = obs['tiles'].copy()
            for i in range(chow_v - 1, chow_v + 2):
                if i == int(obs['last_tile'][1]):
                    continue
                else:
                    tiles.remove(f'{chow_t}{i}')
            play_tile = self.choose_play(tiles, obs['played_tiles'])
            return Action(player, ActionType.CHOW, f'{chow_tile} {play_tile}')
        return pass_action
    

class RLMahjongBot(BaseMahjongBot):
    @staticmethod
    def choose_play(tiles, playedtitles):
        reward = 0
        playtile = None
        for tile in tiles:
            tiles2 = tiles.copy()
            tiles2.remove(tile)
            tilestuple = tuple(tiles2)
            currentReward = Shanten(tilestuple, playedtitles)
            if currentReward > reward:
                reward = currentReward
                playtile = tile
        return playtile

    def action(self, obs: dict) -> Action:
        
        curTitles = obs['tiles']
        AllTitles = ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9",
                    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
                    "F1", "F2", "F3", "F4",
                    "J1", "J2", "J3"
                    ]
        curTitlesList = [(word, 0) for word in AllTitles]
        for tile in curTitles:
            for i, (word, value) in enumerate(curTitlesList):
                if word == tile:
                    curTitlesList[i] = (word, value + 1)
                    break
        
        playTitle = [obs['last_tile']]
        
        playTitleList = [(word, 0) for word in AllTitles]
        for tile in playTitle:
            # print(tile)
            for i, (word, value) in enumerate(playTitleList):
                # print(word)
                if word == tile:
                    # print(word)
                    playTitleList[i] = (word, value + 1)
                    break
        curValues = [value for word, value in curTitlesList]
        # print()
        seavalues = np.array(list(obs['played_tiles'].values()))
        playValues = [value for word, value in playTitleList]
        input = np.array([])
        input = np.concatenate((input, np.array(curValues), np.array(seavalues), np.array(playValues)))       
        
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(player, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            if last_player != player:
                return pass_action
            else:
                fan = self.check_hu(obs)
                if fan >= 1:
                    return Action(player, ActionType.HU, None) # don't change
                if self.check_kong(obs) and checkangang(input):
                    return Action(player, ActionType.KONG, obs['last_tile'])
                if self.check_meld_kong(obs) and checkbugang(input):
                    return Action(player, ActionType.MELD_KONG, obs['last_tile'])
                play_tile = self.choose_play(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])
                return Action(player, ActionType.PLAY, play_tile)

        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        if last_player == player: 
            return pass_action

        fan = self.check_hu(obs)
        if fan >= 1:
            return Action(player, ActionType.HU, None)
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
        if self.check_kong(obs) and checkgang(input):
            return Action(player, ActionType.KONG, None)
        if self.check_pung(obs) and checkpeng:
            tiles = obs['tiles'].copy()
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            play_tile = self.choose_play(tiles, obs['played_tiles'])
            return Action(player, ActionType.PUNG, play_tile)

        chow_list = self.check_chow(obs)
        if checkchi(input):
            if len(chow_list) != 0:
                chow_tile = random.choice(chow_list)
                chow_t, chow_v = chow_tile[0], int(chow_tile[1])
                tiles = obs['tiles'].copy()
                for i in range(chow_v - 1, chow_v + 2):
                    if i == int(obs['last_tile'][1]):
                        continue
                    else:
                        tiles.remove(f'{chow_t}{i}')
                play_tile = self.choose_play(tiles, obs['played_tiles'])
                return Action(player, ActionType.CHOW, f'{chow_tile} {play_tile}')
        return pass_action



# In tryModel.py



class ClaimingModel(nn.Module):
    def __init__(self):
        super(ClaimingModel, self).__init__()
        self.inputSize = 102
        self.outputSize = 2
        self.hiddenSize1 = 128
        self.hiddenSize2 = 128
        
        self.network = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.outputSize)
            # Removed Softmax
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)

    def forward(self, input):
        return self.network(input)
    


def checkchi(input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClaimingModel().to(device)
    model.load_state_dict(torch.load('RL/model/chi.pth'))
    model.eval()
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output)
        return predicted.item()
        # print(f'Predicted class: {predicted.item()}')


def checkpeng(input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClaimingModel().to(device)
    model.load_state_dict(torch.load('RL/model/peng.pth'))
    model.eval()
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output)
        return predicted.item()
        # print(f'Predicted class: {predicted.item()}')

def checkangang(input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClaimingModel().to(device)
    model.load_state_dict(torch.load('RL/model/angang.pth'))
    model.eval()
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output)
        return predicted.item()
        # print(f'Predicted class: {predicted.item()}')
        
def checkbugang(input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClaimingModel().to(device)
    model.load_state_dict(torch.load('RL/model/bugang.pth'))
    model.eval()
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output)
        return predicted.item()
        # print(f'Predicted class: {predicted.item()}')
        
def checkgang(input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClaimingModel().to(device)
    model.load_state_dict(torch.load('RL/model/gang.pth'))
    model.eval()
    input_tensor = torch.tensor(input, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output)
        return predicted.item()
        # print(f'Predicted class: {predicted.item()}')