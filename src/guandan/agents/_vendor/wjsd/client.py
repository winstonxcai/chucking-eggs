# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:

import numpy as np
from .state import State


class WjsdClient(object):

    def __init__(self):
        self.state = State()
        self.shoupaijuzhen = np.zeros(shape=(4,16),dtype='i1')

    def opened(self):
        pass

    def closed(self, code, reason=None):
        pass  #print("Closed down", code, reason)

    def received_message(self, message):
        self.state.parse(message)           # 调用状态对象来解析状态

        pass  #print('当前收到信息是',message)

        if "actionList" in message:         # 需要做出动作选择时调用动作对象进行解析
            rank = self.handsort(message)
            # 同花顺
            straight_flush, flush_i, flush_j = self.tonghuashun()
            # 通配牌同花顺
            straight_tong_flush, flush_tong_i, flush_tong_j = self.straight_tong_flush(rank)
            straight = straight_tong_flush + straight_flush
            # 炸弹
            bomb, bomb_i, bomb_j = self.zhadan(rank)

            zongzha = bomb + straight_flush + straight_tong_flush
            # 顺子
            Straight, straight_i, hangwei, liewei = self.shunzi(rank)
            # 钢板
            TwoTrips, TwoTrips_i, TwoTrips_j = self.gangban()
            # 三对
            ThreePair, ThreePair_i, ThreePair_j = self.sandui()
            # 三张
            Trips, Trips_i, Trips_j = self.trips(rank)
            # 对子
            Pair, Pair_, Pair_j = self.duizi(rank)
            # 单牌
            Single, Single_, Single_j = self.danpai(rank)
            # ThreeWithTwo
            ThreeWithTwo=0
            ThreeWithTwo_i = []
            ThreeWithTwo_j = []
            if Pair > 0 and Trips > 0:
                ThreeWithTwo = Trips * Pair
                for i in range(len(Trips_i)):
                    ThreeWithTwo_i.append(i)
                for j in range(len(Pair_)):
                    ThreeWithTwo_j.append(Pair_[j])


            act_index = 0
            actionlist = message["actionList"]

            action_dict = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A',16:'B',17:'R'}
            hua_dict = {0:'S',1:'H',2:'C',3:'D'}
            # 上贡和还贡
            # stage = self.state._stage
            if message["stage"] == 'tribute':
                if message["type"] == "act":
                    if self.shoupaijuzhen[1][15] == 0 and self.shoupaijuzhen[0][14] == 0:
                        if len(actionlist) > 1 and straight > 0:
                            for i in range(len(flush_j)):
                                for j in range(4):
                                    if self.shoupaijuzhen[j][rank-2] > 0 and (flush_j[i] > self.shoupaijuzhen[j][rank-2] or flush_j[i]+4 < self.shoupaijuzhen[j][rank-2]):
                                        for x in range(len(actionlist)):
                                            if actionlist[x][-1][0][0] == hua_dict[flush_i[i]] and actionlist[x][-1][0][1] == action_dict[flush_j[i]+2]:
                                                act_index = x
                                                pass  #print('自己选择上贡的牌')
                                                return act_index
                            for i in range(len(flush_tong_j)):
                                for j in range(4):
                                    if self.shoupaijuzhen[j][rank-2] > 0 and (flush_tong_j[i] > self.shoupaijuzhen[j][rank-2] or flush_tong_j[i]+4 < self.shoupaijuzhen[j][rank-2]):
                                        for x in range(len(actionlist)):
                                            if actionlist[x][-1][0][0] == hua_dict[flush_tong_i[i]] and actionlist[x][-1][0][1] == action_dict[flush_tong_j[i]+2]:
                                                act_index = x
                                                pass  #print('自己选择上贡的牌')
                                                return act_index
            if message["stage"] == 'back':
                if message["type"] == "act":
                    if Single > 0:
                        pass  #print('还贡单牌选项有', Single_)
                        for i in range(len(Single_)):
                            for j in range(len(actionlist)):
                                if actionlist[j][-1][0][0] == hua_dict[Single_j[i]] and actionlist[j][-1][0][1] == action_dict[Single_[i]+2]:
                                    act_index = j
                                    pass  #print('自己选择还贡的牌')
                                    Single -= 1
                                    return act_index

            myPos = self.state._myPos
            pass  #print('我的位置是：',myPos)
            # public = message["publicInfo"]
            public = self.state._publicInfo
            pass  #print('当前公共剩余牌和打牌信息', public)
            rest_me = public[myPos]['rest']
            greatePos = self.state._greaterPos
            pass  #print('最大动作位置是：',greatePos)
            greateacion = message["greaterAction"]
            pass  #print('当前最大牌信息', greateacion)
            cha = abs(greatePos - myPos)
            row_dict = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14,'B':16,'R':17}

            # 自己先出牌
            actionlist = message["actionList"]
            if greatePos == -1:
                if Straight > 0:
                    for i in range(len(actionlist)):
                        if actionlist[i][0] == 'Straight' and actionlist[i][1] == action_dict[straight_i[0]+2]:
                            pass  #print('自己先出顺子的行和列分别为', hangwei, liewei)
                            if actionlist[i][2][0][0] == hua_dict[hangwei[0]] and actionlist[i][2][0][1] == action_dict[liewei[0]+2] and actionlist[i][2][1][0] == hua_dict[hangwei[1]] and actionlist[i][2][1][1] == action_dict[liewei[1]+2] and  actionlist[i][2][2][0] == hua_dict[hangwei[2]] and actionlist[i][2][2][1] == action_dict[liewei[2]+2] and  actionlist[i][2][3][0] == hua_dict[hangwei[3]] and actionlist[i][2][3][1] == action_dict[liewei[3]+2] and actionlist[i][2][4][0] == hua_dict[hangwei[4]] and actionlist[i][2][4][1] == action_dict[liewei[4]+2]:
                                act_index = i
                                pass  #print('自己先出顺子,剩余通配牌数量是', self.shoupaijuzhen[1][rank-2])
                                Straight -= 1
                                return act_index
                # HR
                elif self.shoupaijuzhen[1][15] > 0 and Single > 0:
                    for j in range(len(Single_)):
                        for i in range(len(actionlist)):
                            if actionlist[i][0] == 'Single' and actionlist[i][1] == action_dict[Single_[j]+2] and actionlist[i][2][0][0] == hua_dict[Single_j[j]]:
                                act_index = i
                                pass  #print('有大王自己先出单牌')
                                Single -= 1
                                return act_index
                elif ThreePair > 0:
                    for j in range(len(ThreePair_i)-5):
                        for i in range(len(actionlist)):
                            if ThreePair_i[j] < 7 or rest_me < 13:
                                if actionlist[i][0] == 'ThreePair' and actionlist[i][1] == action_dict[ThreePair_i[0]+2] and actionlist[i][2][0][0] == hua_dict[ThreePair_j[j]] and actionlist[i][2][1][0] == hua_dict[ThreePair_j[j+1]] and actionlist[i][2][2][0] == hua_dict[ThreePair_j[j+2]] and actionlist[i][2][3][0] == hua_dict[ThreePair_j[j+3]] and actionlist[i][2][4][0] == hua_dict[ThreePair_j[j+4]] and actionlist[i][2][5][0] == hua_dict[ThreePair_j[j+5]]:
                                    act_index = i
                                    pass  #print('自己先出三对')
                                    ThreePair -= 1
                                    return act_index
                elif TwoTrips > 0:
                    for j in range(len(TwoTrips_i)-5):
                        for i in range(len(actionlist)):
                            if TwoTrips_i[j] < 7 or rest_me < 13:
                                if actionlist[i][0] == 'TwoTrips' and actionlist[i][1] == action_dict[TwoTrips_i[0]+2] and actionlist[i][2][0][0] == hua_dict[TwoTrips_j[j]] and actionlist[i][2][1][0] == hua_dict[TwoTrips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[TwoTrips_j[j+2]] and actionlist[i][2][3][0] == hua_dict[TwoTrips_j[j+3]] and actionlist[i][2][4][0] == hua_dict[TwoTrips_j[j+4]] and actionlist[i][2][5][0] == hua_dict[TwoTrips_j[j+5]]:
                                    act_index = i
                                    pass  #print('自己先出钢板')
                                    TwoTrips -= 1
                                    return act_index
                elif (ThreeWithTwo > 0 and Pair_[0]+2 < 11) or (ThreeWithTwo > 0 and rest_me < 12):
                    if Trips_i[0] < 8:
                        for j in range(len(Trips_j)-2):
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'ThreeWithTwo' and actionlist[i][1] == action_dict[Trips_i[j]+2] and actionlist[i][2][-1][-1] == action_dict[Pair_[0]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Trips_j[j]] and actionlist[i][2][1][0] == hua_dict[Trips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[Trips_j[j+2]]:
                                        for x in range(len(Pair_j)-1):
                                            if actionlist[i][2][3][0] == hua_dict[Pair_j[x]] and actionlist[i][2][4][0] == hua_dict[Pair_j[x + 1]]:
                                                act_index = i
                                                pass  #print('自己先出三带二')
                                                Trips -= 1
                                                Pair -= 1
                                                ThreeWithTwo = Trips * Pair
                                                return act_index
                elif Trips > 0:
                    for j in range(len(Trips_j)-2):
                        for i in range(len(actionlist)):
                            if Trips_i[j] < 8 or rest_me < 10:
                                if actionlist[i][0] == 'Trips' and actionlist[i][1] == action_dict[Trips_i[j]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Trips_j[j]] and actionlist[i][2][1][0] == hua_dict[Trips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[Trips_j[j+2]]:
                                        act_index = i
                                        pass  #print('自己先出三张')
                                        Trips -= 1
                                        return act_index
                elif Pair > 0:
                    for j in range(len(Pair_j)-1):
                        for i in range(len(actionlist)):
                            if Pair_j[j] < 7 or rest_me < 9:
                                if actionlist[i][0] == 'Pair' and actionlist[i][1] == action_dict[Pair_[j]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Pair_j[j]] and actionlist[i][2][1][0] == hua_dict[Pair_j[j+1]]:
                                        act_index = i
                                        pass  #print('自己先出对子')
                                        Pair -= 1
                                        return act_index
                elif Single > 0:
                    for j in range(len(Single_)):
                        for i in range(len(actionlist)):
                            if (Single_[j] != 14 and Single_[j] != 15) or rest_me < 8:
                                if actionlist[i][0] == 'Single' and actionlist[i][1] == action_dict[Single_[j]+2] and actionlist[i][2][0][0] == hua_dict[Single_j[j]]:
                                    act_index = i
                                    pass  #print('自己先出单牌')
                                    Single -= 1
                                    return act_index
                elif zongzha > 0:
                    for i in range(len(actionlist)):
                        if actionlist[i][0] == 'Bomb' and rest_me < 10:
                            act_index = i
                            pass  #print('其他牌型都没有自己先出炸弹')
                            Single -= 1
                            return act_index
                else:
                    pass  #print('没有自己先出的牌，随机打出')
                    return act_index

            # 当前队友牌型最大 +++判断单牌和对子是否小于11
            elif cha == 2:
                if greateacion[0] == 'Single' and row_dict[greateacion[1]] < 11 and row_dict[greateacion[1]] != rank and Single > 0:
                    for i in range(len(Single_)):
                        if Single_[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                for x in range(len(Single_j)):
                                    if actionlist[j][0] == 'Single' and actionlist[j][1] == action_dict[Single_[i]+2] and actionlist[j][2][0][0] == hua_dict[Single_j[x]]:
                                        act_index = j
                                        pass  #print('队友出单牌最大出单牌')
                                        Single -= 1
                                        return act_index
                elif greateacion[0] == 'Pair' and Pair > 0 and row_dict[greateacion[1]] < 11 and row_dict[greateacion[1]] != rank:
                    for i in range(len(Pair_)-1):
                        if Pair_[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Pair' and actionlist[j][1] == action_dict[Pair_[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Pair_j[i]] and actionlist[j][2][1][0] == hua_dict[Pair_j[i+1]]:
                                        act_index = j
                                        pass  #print('队友出对子最大出对子')
                                        Pair -= 1
                                        return act_index
                else:
                    pass  #print('队友玩家牌最大，pass')
                    return 0
            # 当前局面对手牌最大
            else:
                if (greateacion[0] == 'StraightFlush' or greateacion[0] == 'Bomb') and zongzha > 0:
                    if len(actionlist) > 1:
                        act_index = 1
                        pass  #print('对手出炸弹或同花顺最大出炸或同花顺')
                        zongzha -= 1
                        return act_index
                # elif greateacion[0] == 'Bomb' and bomb > 0:
                #     if len(actionlist) > 1:
                #         act_index = 1
                #         pass  #print('对手出炸弹最大出炸弹')
                #         bomb -= 1
                #         return act_index
                #         return
                elif greateacion[0] == 'Straight' and Straight > 0:
                    for i in range(len(straight_i)):
                        if straight_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Straight' and actionlist[j][1] == action_dict[straight_i[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[hangwei[i*4+1]] and actionlist[j][2][0][1] == action_dict[liewei[i*4+1]+2] and actionlist[j][2][1][0] == hua_dict[hangwei[i*4+2]] and actionlist[j][2][1][1] == action_dict[liewei[i*4+2]+2] and  actionlist[j][2][2][0] == hua_dict[hangwei[i*4+3]] and actionlist[j][2][2][1] == action_dict[liewei[i*4+3]+2] and  actionlist[j][2][3][0] == hua_dict[hangwei[i*4+4]] and actionlist[j][2][3][1] == action_dict[liewei[i*4+4]+2] and actionlist[j][2][4][0] == hua_dict[hangwei[i*4+5]] and actionlist[j][2][4][1] == action_dict[liewei[i*4+5]+2]:
                                        act_index = j
                                        pass  #print('对手出顺子最大出顺子')
                                        Straight -= 1
                                        return act_index
                elif greateacion[0] == 'ThreePair' and ThreePair > 0:
                    for j in range(len(ThreePair_i)-5):
                        if ThreePair_i[j]+2 > row_dict[greateacion[1]]:
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'ThreePair' and actionlist[i][1] == action_dict[ThreePair_i[j]+2]  and actionlist[i][2][0][0] == hua_dict[ThreePair_j[j]] and actionlist[i][2][1][0] == hua_dict[ThreePair_j[j+1]] and actionlist[i][2][2][0] == hua_dict[ThreePair_j[j+2]] and actionlist[i][2][3][0] == hua_dict[ThreePair_j[j+3]] and actionlist[i][2][4][0] == hua_dict[ThreePair_j[j+4]] and actionlist[i][2][5][0] == hua_dict[ThreePair_j[j+5]]:
                                    act_index = i
                                    pass  #print('队手出三连队最大出三连队')
                                    ThreePair -= 1
                                    return act_index
                elif greateacion[0] == 'TwoTrips' and TwoTrips > 0:
                    for j in range(len(TwoTrips_i)-5):
                        if TwoTrips_i[j]+2 > row_dict[greateacion[1]]:
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'TwoTrips' and actionlist[i][1] == action_dict[TwoTrips_i[j]+2] and actionlist[i][2][0][0] == hua_dict[TwoTrips_j[j]] and actionlist[i][2][1][0] == hua_dict[TwoTrips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[TwoTrips_j[j+2]] and actionlist[i][2][3][0] == hua_dict[TwoTrips_j[j+3]] and actionlist[i][2][4][0] == hua_dict[TwoTrips_j[j+4]] and actionlist[i][2][5][0] == hua_dict[TwoTrips_j[j+5]]:
                                    act_index = i
                                    pass  #print('对手出钢板最大出钢板')
                                    TwoTrips -= 1
                                    return act_index
                elif greateacion[0] == 'Trips' and Trips > 0:
                    for i in range(len(Trips_i)-2):
                        if Trips_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Trips' and actionlist[j][1] == action_dict[Trips_i[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Trips_j[i]] and actionlist[j][2][1][0] == hua_dict[Trips_j[i+1]] and actionlist[j][2][2][0] == hua_dict[Trips_j[i+2]]:
                                        act_index = j
                                        pass  #print('对手出三张最大出三张')
                                        Trips -= 1
                                        return act_index
                elif greateacion[0] == 'ThreeWithTwo' and ThreeWithTwo > 0:
                    for i in range(len(Trips_j)-2):
                        if Trips_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'ThreeWithTwo' and actionlist[j][1] == action_dict[Trips_i[i]+2] and actionlist[j][2][-1][-1] == action_dict[Pair_[0]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Trips_j[i]] and actionlist[j][2][1][0] == hua_dict[Trips_j[i+1]] and actionlist[j][2][2][0] == hua_dict[Trips_j[i+2]]:
                                        for x in range(len(Pair_j)-1):
                                            if actionlist[j][2][3][0] == hua_dict[Pair_j[x]] and actionlist[j][2][4][0] == hua_dict[Pair_j[x + 1]]:
                                                act_index = j
                                                pass  #print('对手出三带二最大出三带二')
                                                Trips -= 1
                                                Pair -= 1
                                                ThreeWithTwo = Trips * Pair
                                                return act_index
                elif zongzha > 3 or (zongzha > 2 and rest_me < 20) or (zongzha > 1 and rest_me < 15) or (zongzha > 0 and rest_me < 10):
                    if len(bomb_i) > 0:
                        for i in range(len(bomb_i)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Bomb' and actionlist[j][1] == action_dict[bomb_i[i]+2]:
                                    act_index = j
                                    pass  #print('对手出的牌型没有出炸弹')
                                    bomb -= 1
                                    return act_index
                    elif len(flush_j) > 0:
                        for i in range(len(flush_j)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'StraightFlush' and actionlist[j][1] == action_dict[flush_j[i]+2]:
                                    act_index = j
                                    pass  #print('对手出的牌型没有出同花顺')
                                    bomb -= 1
                                    return act_index
                    elif len(flush_tong_j) > 0:
                        for i in range(len(flush_tong_j)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'StraightFlush' and actionlist[j][1] == action_dict[flush_tong_j[i]+2]:
                                    act_index = j
                                    pass  #print('对手出的牌型没有出通配同花顺')
                                    bomb -= 1
                                    return act_index
                    elif len(bomb_j) > 0 and rest_me < 9:
                        for j in range(len(bomb_j)):
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'Bomb' and actionlist[i][1] == action_dict[bomb_j[j]+2]:
                                    act_index = i
                                    pass  #print('对手出的牌型没有出6炸上')
                                    bomb -= 1
                                    return act_index
                elif greateacion[0] == 'Pair' and Pair > 0:
                    for i in range(len(Pair_)-1):
                        if Pair_[i]+2 > row_dict[greateacion[1]] or Pair_[i]+2 == rank:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Pair' and actionlist[j][1] == action_dict[Pair_[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Pair_j[i]] and actionlist[j][2][1][0] == hua_dict[Pair_j[i+1]]:
                                        act_index = j
                                        pass  #print('队手出对子最大出对子')
                                        Pair -= 1
                                        return act_index
                elif greateacion[0] == 'Single' and Single > 0:
                    for i in range(len(Single_)):
                        if Single_[i]+2 > row_dict[greateacion[1]] or Single_[i]+2 == rank:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Single' and actionlist[j][1] == action_dict[Single_[i]+2] and actionlist[j][2][0][0] == hua_dict[Single_j[i]]:
                                    act_index = j
                                    pass  #print('对手出单牌最大出单牌')
                                    Single -= 1
                                    return act_index
            pass  #print('自己的判断后没有牌型可以打选择pass 不考虑拆牌')
            return act_index

    def handsort(self, message):
        hand = np.zeros(shape=(4, 16), dtype='i1')
        self.shoupaijuzhen = hand
        shoupai = message["handCards"]
        pass  #print('当前手牌是：',shoupai)
        line_dict = {'S':0,'H':1,'C':2,'D':3}
        row_dict = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14,'B':16,'R':17}
        for i in range(len(shoupai)):
            self.shoupaijuzhen[line_dict[shoupai[i][0]], row_dict[shoupai[i][1]] - 2] += 1

        rank = message["curRank"]
        pass  #print('当前打牌等级是：',row_dict[rank])

        pass  #print('手牌矩阵是\n',self.shoupaijuzhen)
        return row_dict[rank]

    def tonghuashun(self):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('ths',row_sum)
        Single = 0
        Bomb_4 = 0
        StraightFlush=0
        flush_i = []
        flush_j = []
        for i in range(len(row_sum)):
            if row_sum[i]==4:
                Bomb_4 += 1
            if row_sum[i]==1:
                Single += 1
        # 先检验不考虑通配牌时在不拆除4张炸弹的情况下同花顺的数量 或者是拆除4张炸弹单牌数量减小
        for i in range(4):
            for j in range(9):
                if self.shoupaijuzhen[i][j] > 0 and self.shoupaijuzhen[i][j+1] > 0 and self.shoupaijuzhen[i][j+2] > 0 and self.shoupaijuzhen[i][j+3] > 0 and self.shoupaijuzhen[i][j+4] > 0:
                    pass  #print('不考虑通配牌时有同花顺')
                    StraightFlush += 1
                    flush_i.append(i)
                    flush_j.append(j)
                    self.shoupaijuzhen[i][j] -= 1
                    self.shoupaijuzhen[i][j+1] -= 1
                    self.shoupaijuzhen[i][j+2] -= 1
                    self.shoupaijuzhen[i][j+3] -= 1
                    self.shoupaijuzhen[i][j+4] -= 1
        jianshu_i = []
        jianshu_j = []
        if flush_i != []:
            # 判断同花顺是否减少4张炸弹数量和增加的单牌数量
            row_sum_ = self.shoupaijuzhen.sum(axis=0)
            Bomb_4_ = 0
            Single_ = 0
            for i in range(len(row_sum_)):
                if row_sum_[i]==4:
                    Bomb_4_ += 1
                if row_sum_[i]==1:
                    Single_ += 1
            for j in range(len(flush_j)):
                if (Bomb_4 - Bomb_4_ > len(flush_j)-1 and Single - Single_ > 1)  or (Bomb_4 - Bomb_4_ > len(flush_j)):
                    StraightFlush -= 1
                    pass  #print('同花顺拆除了2个炸弹并且单牌比原来多时不考虑组同花顺')
                    self.shoupaijuzhen[flush_i[j]][flush_j[j]] += 1
                    self.shoupaijuzhen[flush_i[j]][flush_j[j]+1] += 1
                    self.shoupaijuzhen[flush_i[j]][flush_j[j]+2] += 1
                    self.shoupaijuzhen[flush_i[j]][flush_j[j]+3] += 1
                    self.shoupaijuzhen[flush_i[j]][flush_j[j]+4] += 1
                    jianshu_i.append(flush_i[j])
                    jianshu_j.append(flush_j[j])
                    # jianshu.append(j)
            for i in range(len(jianshu_i)):
                flush_i.remove(jianshu_i[i])
                flush_j.remove(jianshu_j[i])
        if StraightFlush > 0:
            # for j in range(len(flush_j)):
            #     self.shoupaijuzhen[flush_i[j]][flush_j[j]] -= 1
            #     self.shoupaijuzhen[flush_i[j]][flush_j[j]+1] -= 1
            #     self.shoupaijuzhen[flush_i[j]][flush_j[j]+2] -= 1
            #     self.shoupaijuzhen[flush_i[j]][flush_j[j]+3] -= 1
            #     self.shoupaijuzhen[flush_i[j]][flush_j[j]+4] -= 1
            pass  #print('去除同花顺后更新的手牌矩阵\n', self.shoupaijuzhen)
            pass  #print('ths',row_sum)
            pass  #print('不考虑通配牌的同花顺有:', flush_i, flush_j)
            return StraightFlush, flush_i, flush_j
        else:
            return 0, [], []

    def straight_tong_flush(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('stf',row_sum)
        Single = 0
        Bomb_4 = 0
        for i in range(len(row_sum)):
            if row_sum[i]==4:
                Bomb_4 += 1
            if row_sum[i]==1:
                Single += 1

        Straight_tong_flush = 0
        flush_tong_i = []
        flush_tong_j = []
        dingwei = 0
        if self.shoupaijuzhen[1][rank-2] > 0:
            pass  #print('存在{}张通配牌'.format(self.shoupaijuzhen[1][rank-2]))
            # 存在通配牌先检验不减少炸弹数量的情况下能否添加同花顺
            # row_sum_[num] -= tongpei[0][num]
            pass  #print('考虑通配牌时只考虑单牌有同花顺')
            for i in range(4):
                for j in range(9):
                    if self.shoupaijuzhen[i][j] != self.shoupaijuzhen[1][rank-2]:
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 3] == 1:
                            dingwei = 1
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 3] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 2
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 3] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 3
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 4
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1

        if flush_tong_i != []:
            # 判断同花顺是否减少4张炸弹数量和增加的单牌数量
            row_sum_ = self.shoupaijuzhen.sum(axis=0)
            Bomb_4_ = 0
            Single_ = 0
            for i in range(len(row_sum_)):
                if row_sum_[i]==4:
                    Bomb_4_ += 1
                if row_sum_[i]==1:
                    Single_ += 1
            for j in range(len(flush_tong_j)):
                if (Bomb_4 - Bomb_4_ > len(flush_tong_j)-1 and Single - Single_ > 1) or (Bomb_4 - Bomb_4_ > len(flush_tong_j)):
                    Straight_tong_flush -= 1
                    self.shoupaijuzhen[1][rank-2] += 1
                    pass  #print('同花顺拆除了2个炸弹并且单牌比原来多时不考虑组同花顺')
                    if dingwei == 1:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 2:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 3:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 4:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])

        # 输出同花顺后的手牌
        if Straight_tong_flush > 0:
            pass  #print('tongpei同花顺更新后的手牌矩阵为\n',self.shoupaijuzhen)
            pass  #print('stf',row_sum)
            pass  #print('含有通配牌的同花顺为：', flush_tong_i, flush_tong_j)
            return Straight_tong_flush, flush_tong_i, flush_tong_j
        else:
            return 0, [], []

    def zhadan(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('zd', row_sum)
        # 同花顺后考虑炸弹并更新手牌矩阵
        Bomb = 0
        Bomb_i = []
        Bomb_j = []
        for i in range(len(row_sum)):
            if row_sum[i] == 4:
                Bomb += 1
                Bomb_i.append(i)
                pass  #print('存在炸弹并更新手牌')
                self.shoupaijuzhen[:, [i]] = 0
                row_sum[i] = 0
        # 通配牌的3张配炸弹
        Bomb_tong = []
        if self.shoupaijuzhen[1][rank-2] > 0:
            pass  #print('存在{}张通配牌配3张'.format(self.shoupaijuzhen[1][rank-2]))
            # 检验3张能否添加4张的炸弹
            for i in range(12, -1, -1):
                if self.shoupaijuzhen[1][rank-2] > 0:
                    if row_sum[i] == 3:
                        self.shoupaijuzhen[1][rank-2] -= 1
                        self.shoupaijuzhen[:, [i]] = 0
                        row_sum[i] = 0
                        Bomb += 1
                        Bomb_i.append(i)
                        Bomb_tong.append(i)
        for i in range(len(row_sum)):
            if row_sum[i] == 5:
                Bomb += 1
                Bomb_i.append(i)
                pass  #print('存在炸弹并更新手牌')
                self.shoupaijuzhen[:, [i]] = 0
                row_sum[i] = 0
        for i in range(len(row_sum)):
            if row_sum[i] > 5:
                Bomb += 1
                Bomb_j.append(i)
                pass  #print('存在炸弹并更新手牌')
                self.shoupaijuzhen[:, [i]] = 0
                row_sum[i] = 0
        # 输出炸弹后的手牌
        if Bomb > 0:
            pass  #print('4炸弹通配牌4炸5炸为{}，通配牌炸弹为{}，6炸及以上为{}'.format(Bomb_i, Bomb_tong, Bomb_j))
            pass  #print('炸弹更新后的手牌矩阵为\n', self.shoupaijuzhen)
            pass  #print('zd', row_sum)
            return Bomb, Bomb_i, Bomb_j
        else:
            return 0, [], []

    def shunzi(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('sz', row_sum)
        Straight=0
        Straight_i = []
        Straight_j = []
        hangwei = []
        liewei = []
        dingwei = 0
        for i in range(9):
            if row_sum[i] != 0 and row_sum[i+1] == 1 and row_sum[i+2] == 1 and row_sum[i+3] == 1 and row_sum[i+4] == 1:
                dingwei = 1
                Straight += 1
                Straight_i.append(i)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] > 0:
                        hangwei.append(j)
                        liewei.append(i)
                        self.shoupaijuzhen[j][i] -= 1
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        hangwei.append(j)
                        liewei.append(i+1)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 1:
                        hangwei.append(j)
                        liewei.append(i+2)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+3] == 1:
                        hangwei.append(j)
                        liewei.append(i+3)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+4] == 1:
                        hangwei.append(j)
                        liewei.append(i+4)
                        break
                self.shoupaijuzhen[:,[i+1,i+2,i+3,i+4]] = 0
                row_sum = self.shoupaijuzhen.sum(axis=0)
            if row_sum[i] == 1 and row_sum[i+1] != 0 and row_sum[i+2] == 1 and row_sum[i+3] == 1 and row_sum[i+4] == 1:
                Straight += 1
                Straight_i.append(i)
                dingwei = 2
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        hangwei.append(j)
                        liewei.append(i)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] > 0:
                        hangwei.append(j)
                        liewei.append(i+1)
                        self.shoupaijuzhen[j][i+1] -= 1
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 1:
                        hangwei.append(j)
                        liewei.append(i+2)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+3] == 1:
                        hangwei.append(j)
                        liewei.append(i+3)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+4] == 1:
                        hangwei.append(j)
                        liewei.append(i+4)
                        break
                self.shoupaijuzhen[:,[i,i+2,i+3,i+4]] = 0
                row_sum = self.shoupaijuzhen.sum(axis=0)
            if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+2] != 0 and row_sum[i+3] == 1 and row_sum[i+4] == 1:
                Straight += 1
                Straight_i.append(i)
                dingwei = 3
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        hangwei.append(j)
                        liewei.append(i)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        hangwei.append(j)
                        liewei.append(i+1)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] > 0:
                        hangwei.append(j)
                        liewei.append(i+2)
                        self.shoupaijuzhen[j][i+2] -= 1
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+3] == 1:
                        hangwei.append(j)
                        liewei.append(i+3)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+4] == 1:
                        hangwei.append(j)
                        liewei.append(i+4)
                        break
                self.shoupaijuzhen[:,[i,i+1,i+3,i+4]] = 0
                row_sum = self.shoupaijuzhen.sum(axis=0)
            if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+2] == 1 and row_sum[i+3] != 0 and row_sum[i+4] == 1:
                Straight += 1
                Straight_i.append(i)
                dingwei = 4
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        hangwei.append(j)
                        liewei.append(i)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        hangwei.append(j)
                        liewei.append(i+1)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 1:
                        hangwei.append(j)
                        liewei.append(i+2)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+3] > 0:
                        hangwei.append(j)
                        liewei.append(i+3)
                        self.shoupaijuzhen[j][i+3] -= 1
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+4] == 1:
                        hangwei.append(j)
                        liewei.append(i+4)
                        break
                self.shoupaijuzhen[:,[i,i+1,i+2,i+4]] = 0
                row_sum = self.shoupaijuzhen.sum(axis=0)
            if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+2] == 1 and row_sum[i+3] == 1 and row_sum[i+4] != 0:
                dingwei = 5
                Straight += 1
                Straight_i.append(i)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        hangwei.append(j)
                        liewei.append(i)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        hangwei.append(j)
                        liewei.append(i+1)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 1:
                        hangwei.append(j)
                        liewei.append(i+2)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+3] == 1:
                        hangwei.append(j)
                        liewei.append(i+3)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i+4] > 0:
                        hangwei.append(j)
                        liewei.append(i+4)
                        self.shoupaijuzhen[j][i+4] -= 1
                        break
                self.shoupaijuzhen[:,[i,i+1,i+2,i+3]] = 0
                row_sum = self.shoupaijuzhen.sum(axis=0)
        # 通配牌顺子
        hang_wei = []
        lie_wei = []
        Straight_ = 0
        if self.shoupaijuzhen[1][rank-2] > 0:
            for i in range(9):
                if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+2] == 1 and row_sum[i+3] == 1:
                    Straight_ += 1
                    Straight_j.append(i)
                    dingwei = 5
                    for j in range(4):
                        if self.shoupaijuzhen[j][i] == 1:
                            hang_wei.append(j)
                            lie_wei.append(i)
                            break
                    for j in range(4):
                        if self.shoupaijuzhen[j][i+1] == 1:
                            hang_wei.append(j)
                            lie_wei.append(i+1)
                            break
                    for j in range(4):
                        if self.shoupaijuzhen[j][i+2] == 1:
                            hang_wei.append(j)
                            lie_wei.append(i+2)
                            break
                    for j in range(4):
                        if self.shoupaijuzhen[j][i+3] == 1:
                            hang_wei.append(j)
                            lie_wei.append(i+3)
                            break
                    self.shoupaijuzhen[:,[i,i+1,i+2,i+3]] = 0
                    self.shoupaijuzhen[1][rank-2] -= 1
                    hangwei.append(1)
                    liewei.append(rank-2)
                    # for j in range(4):
                    #     if self.shoupaijuzhen[j][i+4] > 0:
                    #         self.shoupaijuzhen[j][i+4] -= 1
                    #         break
                    row_sum = self.shoupaijuzhen.sum(axis=0)
                # if self.shoupaijuzhen[1][rank - 2] > 0:
                #     if row_sum[i] == 1 and row_sum[i+2] == 1 and row_sum[i+3] == 1 and row_sum[i+4] == 1:
                #         Straight += 1
                #         Straight_i.append(i)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i)
                #                 break
                #         hangwei.append(1)
                #         liewei.append(rank-2)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+2] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+2)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+3] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+3)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+4] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+4)
                #                 break
                #         self.shoupaijuzhen[:,[i,i+2,i+3,i+4]] = 0
                #         self.shoupaijuzhen[1][rank-2] -= 1
                #         row_sum = self.shoupaijuzhen.sum(axis=0)
                # if self.shoupaijuzhen[1][rank-2] > 0:
                #     if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+3] == 1 and row_sum[i+4] == 1:
                #         Straight += 1
                #         Straight_i.append(i)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+1] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+1)
                #                 break
                #         hangwei.append(1)
                #         liewei.append(rank-2)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+3] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+3)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+4] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+4)
                #                 break
                #         self.shoupaijuzhen[:,[i,i+1,i+3,i+4]] = 0
                #         self.shoupaijuzhen[1][rank-2] -= 1
                #         # for j in range(4):
                #         #     if self.shoupaijuzhen[j][i+2] > 0:
                #         #         self.shoupaijuzhen[j][i+2] -= 1
                #         #         break
                #         row_sum = self.shoupaijuzhen.sum(axis=0)
                # if self.shoupaijuzhen[1][rank-2] > 0:
                #     if row_sum[i] == 1 and row_sum[i+1] == 1 and row_sum[i+2] == 1 and row_sum[i+4] == 1:
                #         Straight += 1
                #         Straight_i.append(i)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+1] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+1)
                #                 break
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+2] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+2)
                #                 break
                #         hangwei.append(1)
                #         liewei.append(rank-2)
                #         for j in range(4):
                #             if self.shoupaijuzhen[j][i+4] == 1:
                #                 hangwei.append(j)
                #                 liewei.append(i+4)
                #                 break
                #         self.shoupaijuzhen[:,[i,i+1,i+2,i+4]] = 0
                #         self.shoupaijuzhen[1][rank-2] -= 1
                #         # for j in range(4):
                #         #     if self.shoupaijuzhen[j][i+3] > 0:
                #         #         self.shoupaijuzhen[j][i+3] -= 1
                #         #         break
                #         row_sum = self.shoupaijuzhen.sum(axis=0)
        if Straight > 0:
            pass  #print('顺子为：',hangwei, liewei)
            pass  #print('Straight更新后的手牌矩阵为\n',self.shoupaijuzhen)
            pass  #print('sz', row_sum)
            return Straight, Straight_i, hangwei, liewei
        else:
            return 0, 0, [], []

    def gangban(self):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('gb', row_sum)
        TwoTrips=0
        TwoTrips_i = []
        TwoTrips_j = []
        for i in range(12):
            if row_sum[i] == 3 and row_sum[i+1] == 3:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 3:
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        TwoTrips_j.append(j)
                    if self.shoupaijuzhen[j][i] == 2:
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 3:
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        TwoTrips_j.append(j)
                    if self.shoupaijuzhen[j][i+1] == 2:
                        TwoTrips_j.append(j)
                        TwoTrips_j.append(j)
                TwoTrips += 1
                TwoTrips_i.append(i)
                TwoTrips_i.append(i)
                TwoTrips_i.append(i)
                TwoTrips_i.append(i+1)
                TwoTrips_i.append(i+1)
                TwoTrips_i.append(i+1)
                # self.shoupaijuzhen[:,[i,i+1]] = 0
                row_sum[i] = 0
                row_sum[i+1] = 0
        # 输出TwoTrips后的手牌
        if TwoTrips > 0:
            pass  #print('钢板为：',TwoTrips_i, TwoTrips_j)
            pass  #print('TwoTrips更新后的手牌矩阵为\n',self.shoupaijuzhen)
            pass  #print('gb', row_sum)
            return TwoTrips, TwoTrips_i, TwoTrips_j
        else:
            return 0, [], []

    def sandui(self):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('sd', row_sum)
        ThreePair = 0
        ThreePair_i = []
        ThreePair_j = []
        for i in range(11):
            if row_sum[i] == 2 and row_sum[i + 1] == 2 and row_sum[i + 2] == 2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 2:
                        ThreePair_j.append(j)
                        ThreePair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        ThreePair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 2:
                        ThreePair_j.append(j)
                        ThreePair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+1] == 1:
                        ThreePair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 2:
                        ThreePair_j.append(j)
                        ThreePair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i+2] == 1:
                        ThreePair_j.append(j)
                ThreePair += 1
                ThreePair_i.append(i)
                ThreePair_i.append(i)
                ThreePair_i.append(i+1)
                ThreePair_i.append(i+1)
                ThreePair_i.append(i+2)
                ThreePair_i.append(i+2)
                # self.shoupaijuzhen[:, [i, i + 1, i + 2]] = 0
                row_sum[i] = 0
                row_sum[i + 1] = 0
                row_sum[i + 2] = 0
        # 输出ThreePair后的手牌
        if ThreePair > 0:
            pass  #print('三连队为：', ThreePair_i, ThreePair_j)
            pass  #print('ThreePair更新后的手牌矩阵为\n', self.shoupaijuzhen)
            pass  #print('sd', row_sum)
            return ThreePair, ThreePair_i, ThreePair_j
        else:
            return 0, [], []

    def trips(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('sbd', row_sum)
        Trips = 0
        Trips_i = []
        Trips_j = []
        for i in range(12):
            if row_sum[i] == 3 and i != rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 3:
                        Trips_j.append(j)
                        Trips_j.append(j)
                        Trips_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Trips_j.append(j)
                    if self.shoupaijuzhen[j][i] == 2:
                        Trips_j.append(j)
                        Trips_j.append(j)
                Trips += 1
                Trips_i.append(i)
                Trips_i.append(i)
                Trips_i.append(i)
                # self.shoupaijuzhen[:, [i]] = 0
                row_sum[i] = 0
        for i in range(12):
            if row_sum[i] == 3 and i == rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 3:
                        Trips_j.append(j)
                        Trips_j.append(j)
                        Trips_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Trips_j.append(j)
                    if self.shoupaijuzhen[j][i] == 2:
                        Trips_j.append(j)
                        Trips_j.append(j)
                Trips += 1
                Trips_i.append(i)
                Trips_i.append(i)
                Trips_i.append(i)
                # self.shoupaijuzhen[:, [i]] = 0
                row_sum[i] = 0
        if Trips > 0:
            pass  #print('三不带为：', Trips_i, Trips_j)
            pass  #print('Trips更新后的手牌矩阵为\n', self.shoupaijuzhen)
            pass  #print('sbd', row_sum)
            return Trips, Trips_i, Trips_j
        else:
            return 0, [], []

    def duizi(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('dz', row_sum)
        Pair = 0
        Pair_ = []
        Pair_j = []
        for i in range(16):
            if row_sum[i] == 2 and i != rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 2:
                        Pair_j.append(j)
                        Pair_j.append(j)
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Pair_j.append(j)
                Pair += 1
                # self.shoupaijuzhen[:, [i]] = 0
                Pair_.append(i)
                Pair_.append(i)
        for i in range(16):
            if row_sum[i] == 2 and i == rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 2:
                        Pair_j.append(j)
                        Pair_j.append(j)
                        break
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Pair_j.append(j)
                Pair += 1
                # self.shoupaijuzhen[:, [i]] = 0
                Pair_.append(i)
                Pair_.append(i)
        if Pair > 0:
            pass  #print('对子为：', Pair_, Pair_j)
            pass  #print('对子更新后的手牌矩阵为\n', self.shoupaijuzhen)
            pass  #print('dz', row_sum)
            return  Pair, Pair_, Pair_j
        else:
            return 0, [], []

    def danpai(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        pass  #print('dp', row_sum)
        Single = 0
        Single_ = []
        Single_j = []
        for i in range(len(row_sum)):
            if row_sum[i] == 1 and i != rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Single_j.append(j)
                        break
                Single += 1
                # self.shoupaijuzhen[:, [i]] = 0
                Single_.append(i)
        # 通配牌的单子和顺子加到最后
        for i in range(len(row_sum)):
            if row_sum[i] == 1 and i == rank-2:
                for j in range(4):
                    if self.shoupaijuzhen[j][i] == 1:
                        Single_j.append(j)
                        break
                Single += 1
                # self.shoupaijuzhen[:, [i]] = 0
                Single_.append(i)
        if Single > 0:
            pass  #print('单牌为：', Single_, Single_j)
            pass  #print('单牌更新后的手牌矩阵为\n', self.shoupaijuzhen)
            pass  #print('dp', row_sum)
            return Single, Single_, Single_j
        else:
            return 0, [], []


if __name__ == '__main__':
    try:
        # ws = ExampleClient('ws://127.0.0.1:9618/game/gd/client3')
        ws = ExampleClient('ws://192.168.6.119:9618/game/gd/15942316381548325')
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
