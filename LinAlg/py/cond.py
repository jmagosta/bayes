import math

toplevel_x= -77.1

def abs(x):
    if x < 0:
        a = -x
        print(a)
        if a < -100:
            print('A is super small!')
    else:
        a = x
    return a

another_a = abs(toplevel_x)

print(f'a = {another_a}')