import sys, os

read_in_list = []
with open('py/quote.txt', 'rt') as fd:
    for a_line in fd.readlines():
        read_in_list += [a_line[:-1]]

print(read_in_list)

with open('quote_out2.txt', 'at') as fd_out:
     for a_line in read_in_list:
         fd_out.write(f'{a_line}\n')

with open('quote_out3.txt', 'at') as fd_pr:
     for a_line in read_in_list:
         print(a_line, file=fd_pr)