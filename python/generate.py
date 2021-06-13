import os
import random
import string

key_max_bytes = 64
value_digits = 64
nr_iterations = 64

file_dir = os.path.dirname(os.path.realpath('__file__'))
print(file_dir)
colors_dir = os.path.join(file_dir, '..\data\colors.txt')
print(colors_dir)
write_dir = os.path.join(file_dir, '..\data\data.txt')
print(write_dir)

colors_file = open(colors_dir)
colors = colors_file.read()
colors = colors.split('\n')

new_colors = []

for i in range(nr_iterations):
    for c in colors:
        c = c.lower();
        string_len = len(c.encode('utf-8')) 
        fill_len = key_max_bytes - string_len
        if(string_len < key_max_bytes):
            fill = ''.join(random.choices(string.hexdigits, k=fill_len))
            new_colors.append(c + fill)
            data = ''.join(random.choices(string.hexdigits, k=value_digits))
            new_colors.append(data)


write_file = open(write_dir, "w")
for c in new_colors:
    write_file.write(c + '\n')

    
write_file.close()
colors_file.close()


