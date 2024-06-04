from MahjongGB import MahjongShanten
from MahjongGB import ThirteenOrphansShanten    
from MahjongGB import SevenPairsShanten
from MahjongGB import HonorsAndKnittedTilesShanten
from MahjongGB import KnittedStraightShanten
from MahjongGB import RegularShanten


hand = ("W1", "W9", "T1", "T9", "B1", "B7", "F1", "F2", "F3", "F4", "J1", "J2", "J3")
seatitles =  {"W1" : 2, "W2" : 0, "W3" : 1, "W4" : 0, "W5": 0, "W6" : 0, "W7" : 0, "W8" : 0, "W9": 0,
              "T1" : 2, "T2" : 0, "T3" : 1, "T4" : 0, "T5": 0, "T6" : 0, "T7" : 0, "T8" : 0, "T9": 0,
              "B1" : 2, "B2" : 0, "B3" : 1, "B4" : 0, "B5": 0, "B6" : 0, "B7" : 0, "B8" : 0, "B9": 0,
              "J1" : 2, "J2" : 0, "J3" : 1, "J4" : 0, "F1": 0, "F2" : 0, "F3" : 0, "F4" : 0
              }


def remaintitles(useful, seatirles):
    total = 0
    for titles in useful:
        now = seatitles[titles]
        total += (4 - now)
    return total
        
# print(RegularShanten(hand))
def Shanten(hand, seatitles):
    # print("hand")
    # print(hand)
    # regular shanten 1
    RegularShanten(hand)
    # print(hand)
    # print(RegularShanten(hand))
    # print()
    NumberofRegularShanten, UsefulRegularShanten = RegularShanten(hand)
    RegularShantenRemainingTitles = remaintitles(UsefulRegularShanten, seatitles)
    # print("NumberofRegularShanten")
    # print(NumberofRegularShanten)
    # print("RegularShantenRemainingTitles")
    # print(RegularShantenRemainingTitles)
    rewardRegularShanten = (8 - NumberofRegularShanten) + RegularShantenRemainingTitles * 0.01
    # Knitted Straight shanten 2
    if len(hand) != 13:
        return rewardRegularShanten
    if len(hand) == 13:
        KnittedStraightShanten(hand)
        NumberofKnittedStraightShanten, UsefulKnittedStraightShanten = KnittedStraightShanten(hand)
        KnittedStraightShantenRemainingTitles = remaintitles(UsefulKnittedStraightShanten, seatitles)
        rewardKnittedStraightShanten = (8 - NumberofKnittedStraightShanten) + KnittedStraightShantenRemainingTitles * 0.01
        # seven pair shanten 3
        SevenPairsShanten(hand)
        NumberofSevenPairsShanten, UsefulSevenPairsShanten = SevenPairsShanten(hand)
        SevenPairsShantenRemainingTitles = remaintitles(UsefulSevenPairsShanten, seatitles)
        rewardSevenPairsShanten = (8 - NumberofSevenPairsShanten) + 0.1 * SevenPairsShantenRemainingTitles
        # Honors and Knitted Shanten 4
        HonorsAndKnittedTilesShanten(hand)    
        NumberofHonorsAndKnittedTilesShanten, UsefulHonorsAndKnittedTilesShanten = HonorsAndKnittedTilesShanten(hand)
        HonorsAndKnittedTilesShantenRemainingTitles = remaintitles(UsefulHonorsAndKnittedTilesShanten, seatitles)
        rewardHonorsAndKnittedTilesShanten = (8 - NumberofHonorsAndKnittedTilesShanten) + 0.01 * HonorsAndKnittedTilesShantenRemainingTitles
        # Thriteen Orphans Shanten 5
        ThirteenOrphansShanten(hand)    
        NumberofThirteenOrphansShanten, UsefulThirteenOrphansShanten = ThirteenOrphansShanten(hand)
        ThirteenOrphansShantenRemainingTitles = remaintitles(UsefulThirteenOrphansShanten, seatitles)
        rewrdofThirteenOrphansShanten = (8 - NumberofThirteenOrphansShanten) + ThirteenOrphansShantenRemainingTitles * 0.01
        ShantenlList = [NumberofRegularShanten, NumberofKnittedStraightShanten, NumberofSevenPairsShanten, NumberofHonorsAndKnittedTilesShanten, NumberofThirteenOrphansShanten]
        RewardList = [rewardRegularShanten, rewardKnittedStraightShanten, rewardSevenPairsShanten, rewardHonorsAndKnittedTilesShanten, rewrdofThirteenOrphansShanten]
    FinalReward = 0
    ShantenNumber = 0
    index = 0
    ShantenNumber = 100
    for i in range(5):
        # print(ShantenlList[i])
        # print(RewardList[i])
        if ShantenlList[i] < ShantenNumber :
            ShantenNumber = ShantenlList[i]
            FinalReward = RewardList[i]
            index = i
        elif ShantenlList[i] == ShantenNumber:
            if RewardList[i] > RewardList[index]:
                FinalReward = RewardList[i]
            elif RewardList[index] > RewardList[i]:
                FinalReward = RewardList[i]
    # print("final reward")
    # print(FinalReward)
    return FinalReward
            
    # print(MahjongShanten(hand))

    
Shanten(hand, seatitles)