# Things you should never do in python

# eval()
#     Why: Writes self-modifying code, which is hard to predict

msg = 'Surprize!'
eval(f'''def print(k)
    print({msg})''')
print('Hello')

continue

stmt; stmt

global

