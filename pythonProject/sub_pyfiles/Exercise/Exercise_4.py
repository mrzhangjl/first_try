def print_board(board):
    # 绘制井字棋棋盘，用编号占位空位置
    print(
        f"\n {board[0] if board[0] != ' ' else '1'} | {board[1] if board[1] != ' ' else '2'} | {board[2] if board[2] != ' ' else '3'} ")
    print("---+---+---")
    print(
        f" {board[3] if board[3] != ' ' else '4'} | {board[4] if board[4] != ' ' else '5'} | {board[5] if board[5] != ' ' else '6'} ")
    print("---+---+---")
    print(
        f" {board[6] if board[6] != ' ' else '7'} | {board[7] if board[7] != ' ' else '8'} | {board[8] if board[8] != ' ' else '9'} \n")


def check_winner(board):
    # 检查胜负（横、竖、斜向共8种赢法）
    win_patterns = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 横向
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 纵向
        (0, 4, 8), (2, 4, 6)  # 斜向
    ]
    for a, b, c in win_patterns:
        if board[a] == board[b] == board[c] != ' ':
            return board[a]  # 返回获胜方（X/O）
    return None  # 无胜者


def is_board_full(board):
    # 检查棋盘是否下满（平局）
    return ' ' not in board


def tic_tac_toe():
    board = [' '] * 9  # 初始化棋盘（9个空格）
    current_player = 'X'  # 先手为X
    game_over = False

    print("欢迎玩井字棋！输入1-9对应格子落子（如下）：")
    print_board(board)  # 显示初始棋盘（带编号）

    while not game_over:
        # 接收用户输入
        move = input(f"玩家 {current_player}，请输入落子位置（1-9）：")

        # 验证输入合法性
        if not move.isdigit() or int(move) not in range(1, 10):
            print("输入无效！请输入1-9之间的数字。")
            continue
        pos = int(move) - 1  # 转换为列表索引（0-8）

        if board[pos] != ' ':
            print("该位置已被占用！请重新选择。")
            continue

        # 落子并更新棋盘
        board[pos] = current_player
        print_board(board)

        # 检查游戏结束条件
        winner = check_winner(board)
        if winner:
            print(f"恭喜玩家 {winner} 获胜！🎉")
            game_over = True
        elif is_board_full(board):
            print("棋盘下满，平局！🤝")
            game_over = True
        else:
            # 切换玩家
            current_player = 'O' if current_player == 'X' else 'X'


if __name__ == "__main__":
    tic_tac_toe()