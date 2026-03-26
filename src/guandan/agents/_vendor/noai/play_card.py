# PORTED: Vendored from 2nd Prize Fudan entry (Chen Yuguan)
# Changes: import paths, Strategy passed as argument instead of global singleton,
#          removed print statements

from .create_action_list import CreateActionList
from .count_value import CountValue
from .config import CompareRank
from . import config


class PlayCard():

    def actBack(self, handCards, curRank):
        bestPlay = []
        maxValue = -100
        for rank in config.cardRanks:
            if (rank!=curRank and rank<='9' and rank>='2'):
                for card in handCards:
                    if (card[1]==rank):
                        action = [card]
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        if (restValue>maxValue):
                            maxValue = restValue
                            bestPlay = {"action": action, "type": "back", "rank": rank}
                        break
        return bestPlay

    def GetAdditionalActionList(self, typeList, curRank, fullActionList):
        additionalActionList=[]
        dict_seen = {}
        for action in fullActionList:
            if (action[0] in typeList and ((action[0], action[1]) not in dict_seen.keys())):
                for card in action[2]:
                    if card == 'H'+curRank:
                        additionalActionList.append(action)
                        dict_seen[(action[0], action[1])] = 1
                        break
        return additionalActionList

    def FreePlay(self, handCards, curRank, strategy, fullActionList=None):
        # PORTED: Strategy passed as argument instead of global
        handValue, handActions = CountValue().HandCardsValue(handCards, 0, curRank)
        strategy.SetRole(handValue, handActions, curRank)
        strategy.makeReviseValues()
        additionalActionList = self.GetAdditionalActionList(["ThreePair", "Straight"], curRank, fullActionList) if fullActionList else []

        bestPlay = {}
        if (len(handCards)>=15 or strategy.roundStage != 'ending'):
            minValue = 100
            for action in handActions:
                actionValue = CountValue().ActionValue(action, action['type'], action['rank'], curRank) - strategy.freeActionRV[action['type']] \
                        - strategy.freeActionRV[(action['type'],action['rank'])]
                if actionValue < minValue:
                    minValue = actionValue
                    bestPlay = action
        else:
            maxValue = -100
            actionList = CreateActionList().CreateList(handCards)
            for i in range(0, len(config.cardTypes)):
                type = config.cardTypes[i]
                for rank1 in actionList[type]:
                    for card in actionList[type][rank1]:
                        color = None
                        rank = rank1
                        if (type == 'StraightFlush'):
                            rank = rank1[1]
                            color = rank1[0]
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += strategy.freeActionRV[type]
                        if (type, rank) in strategy.freeActionRV.keys():
                            thisHandValue += strategy.freeActionRV[(type, rank)]
                        if (thisHandValue < 0): thisHandValue = 0
                        if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and \
                            (bestPlay == {} or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = thisHandValue + restValue
                            bestPlay = {"action": action, "type": type, "rank": rank}

            # try additional list
            for action in additionalActionList:
                type = action[0]
                rank = action[1]
                card = rank
                if type == 'Bomb':
                    card = len(action[2])

                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                restValue += strategy.handRV.get(type, 0)
                thisHandValue = CountValue().ActionValue(action[2], type, rank, curRank)
                thisHandValue += strategy.freeActionRV[type]
                if (type, rank) in strategy.freeActionRV.keys():
                    thisHandValue += strategy.freeActionRV[(type, rank)]
                if (thisHandValue < 0): thisHandValue = 0
                if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and
                                (bestPlay == {} or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = thisHandValue + restValue
                    bestPlay = {"action": action[2], "type": type, "rank": rank}

        return bestPlay

    def RestrictedPlay(self, handCards, formerAction, curRank, strategy, fullActionList=None):
        # PORTED: Strategy passed as argument instead of global
        actionList = CreateActionList().CreateList(handCards)

        additionalActionList = self.GetAdditionalActionList(["Bomb", "StraightFlush", "ThreePair", "Straight"], curRank,
                                                            fullActionList) if fullActionList else []

        bestPlay = []
        maxValue, restActions = CountValue().HandCardsValue(handCards, 0, curRank)
        strategy.SetRole(maxValue, restActions, curRank)
        strategy.makeReviseValues()
        maxValue += strategy.restrictedActionRV.get("PASS", 0)

        for i in range(0, len(config.cardTypes)):
            type = config.cardTypes[i]
            if (type != 'Bomb' and type != 'StraightFlush' and type != formerAction["type"]): continue
            for rank1 in actionList[type]:
                for card in actionList[type][rank1]:
                    color = None
                    rank = rank1
                    if (type == 'StraightFlush'):
                        rank = rank1[1]
                        color = rank1[0]
                    if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += strategy.restrictedActionRV[type]
                        if (type, rank) in strategy.restrictedActionRV.keys():
                            thisHandValue += strategy.restrictedActionRV[(type, rank)]
                        if (thisHandValue < 0): thisHandValue = 0
                        if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and \
                        (bestPlay==[] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = thisHandValue + restValue
                            bestPlay = {"action": action, "type": type, "rank": rank}

        # try additional list
        for action in additionalActionList:
            type = action[0]
            rank = action[1]
            card = rank
            if type == 'Bomb':
                card = len(action[2])
            if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                thisHandValue = CountValue().ActionValue(action[2], type, rank, curRank)
                thisHandValue += strategy.restrictedActionRV[type]
                if (type, rank) in strategy.restrictedActionRV.keys():
                    thisHandValue += strategy.restrictedActionRV[(type, rank)]
                if (thisHandValue < 0): thisHandValue = 0
                if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and
                                                (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = thisHandValue + restValue
                    bestPlay = {"action": action[2], "type": type, "rank": rank}

        if (bestPlay==[]):
            bestPlay = {'action': 'PASS', 'type': 'PASS', 'rank': 'PASS'}
        return bestPlay
