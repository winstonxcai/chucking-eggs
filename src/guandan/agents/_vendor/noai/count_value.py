# PORTED: Vendored from 2nd Prize Fudan entry (Chen Yuguan)
# Changes: import path only

from .create_action_list import CreateActionList
from . import config


class CountValue():
    def ActionValue(self, action, type, rank, curRank):
        value=0
        if len(action) == 0: return 0
        if type == 'Single':
            if (rank == 'R'):
                value=0.5
            elif (rank == 'B'):
                value=0
            elif (rank == curRank):
                value=-0.5
            else :
                value=-1
        elif type == 'Pair':
            if (rank == 'R'):
                value=1
            elif (rank == 'B'):
                value=0.5
            elif (rank == curRank or rank == 'A'):
                value=0
            elif (rank == 'K'):
                value=-0.5
            else :
                value=-1
        elif (type == 'Trips' or type == 'ThreeWithTwo'):
            if (rank == curRank):
                value=0.5
            elif (rank == 'A' or rank == 'K'):
                value=0
            elif (rank == 'Q' or rank == 'J' or rank == 'T'):
                value=-0.5
            else:
                value=-1
        elif (type == 'TwoTrips' or type =='ThreePair'):
            value = -0.5
        elif (type == 'Straight'):
            if (rank == '8' or rank=='9' or rank=='T'):
                value = -0.5
            else:
                value = -1
        elif (type == 'Bomb'):
            value = 1
        elif (type == 'StraightFlush'):
            value = 1

        return value

    def OnlyPairAndSingleHandValue(self, handCards, curRank):
        retValue=0
        retActions=[]
        handCards.sort(key=lambda card: card[1])
        p=0
        while p<len(handCards):
            if (p<len(handCards)-1 and handCards[p][1]==handCards[p+1][1]):
                action = [handCards[p],handCards[p+1]]
                retValue +=self.ActionValue(action, 'Pair', handCards[p][1], curRank)
                retActions.append({'action': action, 'type': 'Pair', 'rank': handCards[p][1]})
                p+=2
            else:
                action = [handCards[p]]
                retValue +=self.ActionValue([handCards[p]], 'Single', handCards[p][1], curRank)
                retActions.append({'action': action, 'type': 'Single', 'rank': handCards[p][1]})
                p+=1
        return retValue, retActions

    def OnlyTripsAndPairAndSingleHandValue(self, handCards, curRank):
        retValue=0
        retActions=[]
        handCards.sort(key=lambda card: card[1])
        p=0
        while p<len(handCards):
            if (p<len(handCards)-2 and handCards[p][1]==handCards[p+1][1] and handCards[p+1][1]==handCards[p+2][1]):
                action = [handCards[p],handCards[p+1],handCards[p+2]]
                retValue += self.ActionValue(action, 'Trips', handCards[p][1], curRank)
                retActions.append({'action': action, 'type': 'Trips', 'rank': handCards[p][1]})
                p+=3
            elif (p<len(handCards)-1 and handCards[p][1]==handCards[p+1][1]):
                action = [handCards[p],handCards[p+1]]
                retValue += self.ActionValue(action, 'Pair', handCards[p][1], curRank)
                retActions.append({'action': action, 'type': 'Pair', 'rank': handCards[p][1]})
                p+=2
            else:
                action = [handCards[p]]
                retValue += self.ActionValue([handCards[p]], 'Single', handCards[p][1], curRank)
                retActions.append({'action': action, 'type': 'Single', 'rank': handCards[p][1]})
                p+=1
        return retValue, retActions

    def GetCountFromHand(self, handCards):
        countCards = {}
        for card in handCards:
            if card[1] not in countCards.keys():
                countCards[card[1]] = 1
            else:
                countCards[card[1]] += 1
        return countCards

    def HandCardsValue(self, handCards, nowType, curRank, initRank='2'):
        if len(handCards)==0: return 0,[]
        if nowType >= config.cardTypes.index('Trips'):
            return self.OnlyTripsAndPairAndSingleHandValue(handCards, curRank)
        actionList = CreateActionList().CreateList(handCards)
        countCards = self.GetCountFromHand(handCards)

        bestActions=[]
        maxValue=-100
        nowRank=initRank
        for i in range(nowType, len(config.cardTypes)):
            type = config.cardTypes[i]
            for rank1 in actionList[type]:
                color = None
                rank = rank1
                if (type == 'StraightFlush'):
                    rank = rank1[1]
                    color = rank1[0]
                if config.cardRanks.index(rank)<config.cardRanks.index(nowRank):
                    continue
                for card in actionList[type][rank1]:
                    if (type == 'ThreeWithTwo'):
                        if countCards[card]!=2: continue
                        if card == curRank: continue
                    action = CreateActionList().GetAction(type, rank, card, handCards, color)
                    restCards = CreateActionList().GetRestCards(action, handCards)
                    thisHandValue = restValue = 0
                    thisHandValue = self.ActionValue(action, type, rank, curRank)
                    restValue, restActions = self.HandCardsValue(restCards, i, curRank, initRank)
                    if (thisHandValue + restValue > maxValue):
                        maxValue = thisHandValue + restValue
                        bestActions = [{'action': action, 'type': type, 'rank': rank}] + restActions
                    if (type=='ThreeWithTwo'):
                        break
            nowRank = '2'
        return maxValue, bestActions
