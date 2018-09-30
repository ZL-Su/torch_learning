import numpy as np

the_world_is_flat = True
if the_world_is_flat:
   print("Be careful not to fall off!")

squares = [1, 4, 9, 16, 25]
print(squares)

slice_squares = squares[-3:]
print(slice_squares)

x = int(input("Please enter an integer: "))
if x < 0:
   x = 0;
   print('Negative changed to zero')
elif x == 0:
   print('Zero')
elif x == 1:
   print('Single')
else:
   print('More...')

words = ['cat', 'window', 'defenestrate']
for w in words:
   print(w, len(w))
for w in words[:]:
   if(len(w) > 6):
      words.insert(0, w)

print(words)
