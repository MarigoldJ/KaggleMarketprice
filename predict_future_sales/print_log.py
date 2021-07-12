import sys

sys.stdout = open('./output.txt', 'w')

print('hello world!!')
print()
print('no')

sys.stdout.close()
