from start_game.keyboard_operation import Operation
import time





def NextGame():
    # esc
    Operation('esc')
    time.sleep(3)
    # 捡装备
    Operation('0')
    time.sleep(2)
    # 卖装备
    Operation('a')

    # 再次挑战
    Operation('f10')
    time.sleep(3)

if __name__ == '__main__':
    NextGame()

    pass