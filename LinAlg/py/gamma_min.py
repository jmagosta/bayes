import math
import sys
print(sys.version_info)

start_x = 1.0
end_x = 2.0

min_step = 1e-3
step_size = 0.05
epsilon = 1e-3

def still_decreasing(previous_gamma, new_gamma):
    'Is gamma estimate still decreasing? '
    step = previous_gamma - new_gamma
    print(f'decreased by {step}')
    return step > epsilon

last_x = start_x
next_x = last_x + step_size
current_guess = math.gamma(last_x)
next_guess = math.gamma(next_x)

while still_decreasing(current_guess, next_guess):
    last_x = next_x 
    next_x = last_x + step_size
    current_guess = math.gamma(last_x)
    next_guess = math.gamma(next_x)

print(f'x = {last_x}, gamma = {current_guess}')

# See this referene https://www.nature.com/articles/135917b0