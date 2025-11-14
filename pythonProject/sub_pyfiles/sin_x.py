import math


def sin(x):
    answer = x
    symbol = False
    for i in range(3, 171, 2):
        answer += (x**i / math.factorial(i)
                   ) if symbol else (-x**i / math.factorial(i))
        symbol = not symbol
    return answer


if __name__ == '__main__':
    result = sin(math.pi / 6)
    print('result =', result)
    print('result =', math.sin(math.pi / 6))


