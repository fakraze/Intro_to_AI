
import random
from collections import Counter
from typing import List
from .consts import ActionType, ClaimingType
from .player_data import Action
from MahjongGB import MahjongFanCalculator
from MahjongGB import RegularShanten
import copy
from .ten import *
import torch
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mahjong_env.nn import *

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

# class AstarMahjongBot(BaseMahjongBot):
#     @staticmethod
#     def choose_play(tiles, playedtitles):
#         reward = 0
#         playtile = None
#         for tile in tiles:
#             tiles2 = tiles.copy()
#             tiles2.remove(tile)
#             tilestuple = tuple(tiles2)
#             currentReward = Shanten(tilestuple, playedtitles)
#             if currentReward > reward:
#                 reward = currentReward
#                 playtile = tile
#             # print("tile")
#             # print(tile)
#             # print("currentreward")
#             # print(currentReward )
#         return playtile
        

#     def action(self, obs: dict) -> Action:
#         if len(obs) == 0:
#             return Action(0, ActionType.PASS, None)
#         player = obs['player_id']
#         last_player = obs['last_player']
#         pass_action = Action(player, ActionType.PASS, None)

#         if obs['last_operation'] == ActionType.DRAW:
#             if last_player != player:
#                 return pass_action
#             else:
#                 fan = self.check_hu(obs)
#                 if fan >= 1:
#                     return Action(player, ActionType.HU, None)
#                 if self.check_kong(obs):
#                     return Action(player, ActionType.KONG, obs['last_tile'])
#                 if self.check_meld_kong(obs):
#                     return Action(player, ActionType.MELD_KONG, obs['last_tile'])
#                 # print("obs['tiles']")
#                 # print(obs['tiles'])
#                 # print("[obs['last_tile']]")
#                 # print([obs['last_tile']])
#                 play_tile = self.choose_play(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])
#                 return Action(player, ActionType.PLAY, play_tile)

#         if obs['last_operation'] == ActionType.KONG:
#             return pass_action
#         if last_player == player:
#             return pass_action

#         fan = self.check_hu(obs)
#         if fan >= 1:
            
#             return Action(player, ActionType.HU, None)
#         if obs['last_operation'] == ActionType.MELD_KONG:
#             return pass_action
#         if self.check_kong(obs):
#             return Action(player, ActionType.KONG, None)
#         if self.check_pung(obs):
#             tiles = obs['tiles'].copy()
#             tiles.remove(obs['last_tile'])
#             tiles.remove(obs['last_tile'])
#             play_tile = self.choose_play(tiles, obs['played_tiles'])
#             return Action(player, ActionType.PUNG, play_tile)

#         chow_list = self.check_chow(obs)
#         if len(chow_list) != 0:
#             chow_tile = random.choice(chow_list)
#             chow_t, chow_v = chow_tile[0], int(chow_tile[1])
#             tiles = obs['tiles'].copy()
#             for i in range(chow_v - 1, chow_v + 2):
#                 if i == int(obs['last_tile'][1]):
#                     continue
#                 else:
#                     tiles.remove(f'{chow_t}{i}')
#             play_tile = self.choose_play(tiles, obs['played_tiles'])
#             return Action(player, ActionType.CHOW, f'{chow_tile} {play_tile}')
#         return pass_action


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
        return playtile, reward
        

    def action(self, obs: dict) -> Action:
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(player, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            # 別人摸牌
            if last_player != player:
                return pass_action
            else:
                # 自己摸牌
                play_tile, playReward = self.choose_play(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])
                
                # 胡
                fan = self.check_hu(obs)
                if fan >= 1:
                    return Action(player, ActionType.HU, None)
                
                # 暗槓
                if self.check_kong(obs):
                    # 不槓 play
                    notKongReward = playReward

                    # 槓             
                    tiles = copy.deepcopy(obs['tiles'])
                    tiles.remove(obs['last_tile'])
                    tiles.remove(obs['last_tile'])
                    tiles.remove(obs['last_tile'])
                    
                    KongReward = Shanten(tuple(tiles), obs['played_tiles'])
                    
                    if KongReward >= notKongReward:
                        return Action(player, ActionType.KONG, obs['last_tile'])
                
                # 補槓
                if self.check_meld_kong(obs):
                    # 不槓 play
                    notKongReward = playReward
                    
                    # 槓
                    KongReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])
                    
                    if KongReward >= notKongReward:
                        return Action(player, ActionType.MELD_KONG, obs['last_tile'])
                
                return Action(player, ActionType.PLAY, play_tile)

        # 槓
        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        # 上一步是自己 且不是摸牌
        if last_player == player:
            return pass_action

        # 別人打牌
        fan = self.check_hu(obs)
        if fan >= 1:
            return Action(player, ActionType.HU, None)
        
        # 補槓
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
        
        # 槓別人
        if self.check_kong(obs):
            # 不槓 pass
            notKongReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])

            # 槓             
            tiles = copy.deepcopy(obs['tiles'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            
            KongReward = Shanten(tuple(tiles), obs['played_tiles'])
            
            if KongReward < notKongReward:
                return pass_action

            return Action(player, ActionType.KONG, None)
        
        # 碰
        if self.check_pung(obs):
            notPungReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])

            tiles = copy.deepcopy(obs['tiles'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            play_tile, pungReward = self.choose_play(tiles, obs['played_tiles'])
            # 如果不碰比較好
            if pungReward < notPungReward:
                return pass_action
            
            return Action(player, ActionType.PUNG, play_tile)

        # 吃
        chow_list = self.check_chow(obs)
        if len(chow_list) != 0:
            maxReward = 0
            maxChow = None
            maxPlay = None

            notEatReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])

            # iterates all chow
            for chow_tile in chow_list:
                chow_t, chow_v = chow_tile[0], int(chow_tile[1])    # int: 一萬 -> W1 -> 1
                tiles = copy.deepcopy(obs['tiles'])
                # 把手上吃的牌 去掉
                for i in range(chow_v - 1, chow_v + 2):
                    if i == int(obs['last_tile'][1]):
                        continue
                    else:
                        tiles.remove(f'{chow_t}{i}')
                
                # iterates all play tile
                for tile in tiles:
                    tiles2 = copy.deepcopy(tiles)
                    tiles2.remove(tile)
                    tilestuple = tuple(tiles2)
                    currentReward = Shanten(tilestuple, obs['played_tiles'])
                    if currentReward > maxReward:
                        maxReward = currentReward
                        maxChow = chow_tile
                        maxPlay = tile

            # 如果不吃比較好
            if maxReward < notEatReward:
                return pass_action
            
            # best chow
            chow_t, chow_v = maxChow[0], int(maxChow[1])    # int: 一萬 -> W1 -> 1
            tiles = obs['tiles'].copy()
            # 把手上吃的牌 去掉
            for i in range(chow_v - 1, chow_v + 2):
                if i == int(obs['last_tile'][1]):
                    continue
                else:
                    tiles.remove(f'{chow_t}{i}')

            # 看要打哪張
            return Action(player, ActionType.CHOW, f'{maxChow} {maxPlay}')
        
        return pass_action


tile_type =  ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9",
              "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
              "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
              "J1", "J2", "J3", "F1", "F2", "F3", "F4"]

class ExpectiMaxMahjongBot(BaseMahjongBot):
    @staticmethod
    def choose_play_by_astar(tiles, playedtitles):
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
        return playtile, reward
    
    def expectimax(tiles, playedtitles, curDepth, depth, remain_tile_cnt):
        if curDepth == depth:
            tilestuple = tuple(tiles)
            return Shanten(tilestuple, playedtitles)

        val_nodes = []
        # 第一次 不摸牌
        if curDepth == 0:   
            for tile in tiles:
                tiles2 = tiles.copy()
                tiles2.remove(tile)
                playedtitles[tile] += 1
                val_nodes.append(ExpectiMaxMahjongBot.expectimax(tiles2, playedtitles, curDepth + 1, depth, remain_tile_cnt))
                playedtitles[tile] -= 1
        # 其他次 計算摸牌機率
        else:
            value = 0
            # 計算摸牌機率
            for get_tile in tile_type:
                if get_tile in playedtitles:
                    remain = 4 - playedtitles[get_tile]
                for handTile in tiles:
                    if handTile == get_tile:
                        remain -= 1
                
                if remain < 0:
                    print("cur tile: ", get_tile)
                    print("played_tiles: ", playedtitles)
                    print("tiles", tiles)
                    print("ERROR")
                    return
                elif remain == 0:
                    continue
                else:
                    prob = remain / remain_tile_cnt
                
                # 加入牌組
                tiles2 = tiles.copy()
                tiles2.append(get_tile)

                # 胡牌
                flag = 1
                NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(tiles))
                if NumberofRegularShanten == 0:
                    for i in UsefulRegularShanten:
                        if i == get_tile:
                            tmp_val = 20
                            flag = 0

                # 沒有胡
                if flag:
                    # 打牌
                    tmp_val = 0
                    for remove_tile in tiles2:
                        tiles3 = tiles2.copy()
                        tiles3.remove(remove_tile)
                        tilestuple = tuple(tiles3)
                        playedtitles[remove_tile] += 1
                        tmp_val = max(tmp_val, ExpectiMaxMahjongBot.expectimax(tiles, playedtitles, curDepth + 1, depth, remain_tile_cnt-1))
                        playedtitles[remove_tile] -= 1

                value += prob * tmp_val
            return value 
    
        # return action
        if curDepth == 0:
            for i in range(len(val_nodes)):
                if val_nodes[i] == max(val_nodes):
                    return tiles[i], val_nodes[i]

    
    @staticmethod
    def choose_play(tiles, playedtitles, initialDepth = 0):
        depth = 2
        # 計算所有可能抽到的牌
        remain_tile_cnt = 34*4
        # 海底
        for tile in tile_type:
            if tile in playedtitles:
                remain_tile_cnt -= playedtitles[tile]
        # 手牌
        remain_tile_cnt -= len(tiles)
        
        return ExpectiMaxMahjongBot.expectimax(tiles, playedtitles, initialDepth, depth, remain_tile_cnt)

    def action(self, obs: dict) -> Action:
        
        if len(obs) == 0:
            return Action(0, ActionType.PASS, None)
        player = obs['player_id']
        last_player = obs['last_player']
        pass_action = Action(player, ActionType.PASS, None)

        if obs['last_operation'] == ActionType.DRAW:
            # 別人摸牌
            if last_player != player:
                return pass_action
            # 自己摸牌
            else:
                # 若向聽數>=3 則用astar 節省時間
                NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(obs['tiles']))
                if NumberofRegularShanten > 2:
                    # Astar
                    play_tile, playReward = self.choose_play_by_astar(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])
                else:
                    # expectiMax
                    play_tile, playReward = self.choose_play(obs['tiles'] + [obs['last_tile']], obs['played_tiles'])

                # 胡
                fan = self.check_hu(obs)
                if fan >= 1:
                    return Action(player, ActionType.HU, None)
                
                # 暗槓
                if self.check_kong(obs):
                    # 不槓 play
                    notKongReward = playReward

                    # 槓             
                    tiles = copy.deepcopy(obs['tiles'])
                    tiles.remove(obs['last_tile'])
                    tiles.remove(obs['last_tile'])
                    tiles.remove(obs['last_tile'])
                    
                    # KongReward = Shanten(tuple(tiles), played_tiles)
                    if NumberofRegularShanten > 2:
                        # Astar
                        KongReward = Shanten(tuple(tiles), obs['played_tiles'])
                    else:
                        KongReward = self.choose_play(tiles, obs['played_tiles'], 1)
                    
                    # KongReward = Shanten(tuple(tiles), obs['played_tiles'])
                    if KongReward >= notKongReward:
                        return Action(player, ActionType.KONG, obs['last_tile'])


                if self.check_meld_kong(obs):
                    # 不槓 play
                    notKongReward = playReward
                    
                    # 槓
                    if NumberofRegularShanten > 2:
                        # Astar
                        KongReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])
                    else:
                        KongReward = self.choose_play(obs['tiles'], obs['played_tiles'], 1)
                    
                    if KongReward >= notKongReward:
                        return Action(player, ActionType.MELD_KONG, obs['last_tile'])
                
                return Action(player, ActionType.PLAY, play_tile)


        # 槓
        if obs['last_operation'] == ActionType.KONG:
            return pass_action
        # 上一步是自己 且不是摸牌
        if last_player == player:
            return pass_action

        # 別人打牌
        fan = self.check_hu(obs)
        if fan >= 1:
            return Action(player, ActionType.HU, None)
        
        # 補槓
        if obs['last_operation'] == ActionType.MELD_KONG:
            return pass_action
    
        # 槓別人
        if self.check_kong(obs):
            # 沒有摸牌
            # 若向聽數>=3 則用astar 節省時間
            NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(obs['tiles']))
            if NumberofRegularShanten > 2:
                # Astar
                curReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])
            else:
                # expectiMax
                curReward = self.choose_play(obs['tiles'], obs['played_tiles'], 1)
            # 不槓 pass
            notKongReward = curReward

            # 槓             
            tiles = copy.deepcopy(obs['tiles'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])
            
            if NumberofRegularShanten > 2:
                KongReward = Shanten(tuple(tiles), obs['played_tiles'])
            else:
                KongReward = self.choose_play(tiles, obs['played_tiles'], 1)

            if KongReward < notKongReward:
                return pass_action

            return Action(player, ActionType.KONG, None)

        
        # 碰
        if self.check_pung(obs):
            # 沒有摸牌
            # 若向聽數>=3 則用astar 節省時間
            NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(obs['tiles']))
            if NumberofRegularShanten > 2:
                # Astar
                curReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])
            else:
                # expectiMax
                curReward = self.choose_play(obs['tiles'], obs['played_tiles'], 1)
            notPungReward = curReward

            tiles = copy.deepcopy(obs['tiles'])
            tiles.remove(obs['last_tile'])
            tiles.remove(obs['last_tile'])

            NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(obs['tiles']))
            # 若向聽數>=3 則用astar 節省時間
            if NumberofRegularShanten > 2:
                play_tile, pungReward = self.choose_play_by_astar(tiles, obs['played_tiles'])
            else:
                play_tile, pungReward = self.choose_play(tiles, obs['played_tiles'])
                
            # 如果不碰比較好
            if pungReward < notPungReward:
                return pass_action
            
            return Action(player, ActionType.PUNG, play_tile)
        
        # 吃
        chow_list = self.check_chow(obs)
        if len(chow_list) != 0:
            maxReward = 0
            maxChow = None
            maxPlay = None

            # 沒有摸牌
            # 若向聽數>=3 則用astar 節省時間
            NumberofRegularShanten, UsefulRegularShanten = RegularShanten(tuple(obs['tiles']))
            if NumberofRegularShanten > 2:
                # Astar
                curReward = Shanten(tuple(obs['tiles']), obs['played_tiles'])
            else:
                # expectiMax
                curReward = self.choose_play(obs['tiles'], obs['played_tiles'], 1)
            notEatReward = curReward

            # iterates all chow
            for chow_tile in chow_list:
                chow_t, chow_v = chow_tile[0], int(chow_tile[1])    # int: 一萬 -> W1 -> 1
                tiles = copy.deepcopy(obs['tiles'])
                # 把手上吃的牌 去掉
                for i in range(chow_v - 1, chow_v + 2):
                    if i == int(obs['last_tile'][1]):
                        continue
                    else:
                        tiles.remove(f'{chow_t}{i}')
                
                # iterates all play tile
                for tile in tiles:
                    tiles2 = copy.deepcopy(tiles)
                    tiles2.remove(tile)
                    tilestuple = tuple(tiles2)
                    
                    # 若向聽數>=3 則用astar 節省時間
                    if NumberofRegularShanten > 2:
                        currentReward = Shanten(tilestuple, obs['played_tiles'])
                    else:
                        currentReward = self.choose_play(tiles2, obs['played_tiles'], 1)
                        
                    if currentReward > maxReward:
                        maxReward = currentReward
                        maxChow = chow_tile
                        maxPlay = tile

            # 如果不吃比較好
            if maxReward < notEatReward:
                return pass_action
            
            # best chow
            chow_t, chow_v = maxChow[0], int(maxChow[1])    # int: 一萬 -> W1 -> 1
            tiles = copy.deepcopy(obs['tiles'])
            # 把手上吃的牌 去掉
            for i in range(chow_v - 1, chow_v + 2):
                if i == int(obs['last_tile'][1]):
                    continue
                else:
                    tiles.remove(f'{chow_t}{i}')

            # 看要打哪張
            return Action(player, ActionType.CHOW, f'{maxChow} {maxPlay}')

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

class NN1gent(BaseMahjongBot):
    def init(self, discardModelPath, chiModelPath, pengModelPath, gangModelPath, bugangModelPath, angangModelPath):

        self.pengModel=NN1Claiming()
        self.pengModel.load_state_dict(torch.load(pengModelPath))
        self.gangModel=NN1Claiming()
        self.gangModel.load_state_dict(torch.load(gangModelPath))
        self.bugangModel=NN1Claiming()
        self.bugangModel.load_state_dict(torch.load(bugangModelPath))
        self.angangModel=NN1Claiming()
        self.angangModel.load_state_dict(torch.load(angangModelPath))
        self.chiModel=NN1Claiming()
        self.chiModel.load_state_dict(torch.load(chiModelPath))
        self.discardModel=NN1Discard()
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
