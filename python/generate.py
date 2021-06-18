import os
import random
import string

key_max_bytes = 64
value_digits = 64
nr_iterations = 64

nr_keys = 32 * 1024;

file_dir = os.path.dirname(os.path.realpath('__file__'))
print(file_dir)
colors_dir = os.path.join(file_dir, '..\data\colors.txt')
print(colors_dir)
write_file_dir = os.path.join(file_dir, '..\data\data_write.txt')
print(write_file_dir)
read_file_dir = os.path.join(file_dir, '..\data\data_read.txt')
print(read_file_dir)

colors_file = open(colors_dir)
colors = colors_file.read()
colors = colors.split('\n')

write_data = []
read_data = []

for i in range(nr_iterations):
    for c in colors:
        c = c.lower();
        string_len = len(c.encode('utf-8')) 
        fill_len = key_max_bytes - string_len
        if(string_len < key_max_bytes):
            fill = ''.join(random.choices(string.hexdigits, k=fill_len))
            write_data.append(c + fill)
            read_data.append(c + fill)
            data = ''.join(random.choices(string.hexdigits, k=value_digits))
            write_data.append(data)


write_data_file = open(write_file_dir, "w")
read_data_file = open(read_file_dir, "w")
for i in range(2*nr_keys):
    write_data_file.write(write_data[i] + '\n')

for i in range(nr_keys):
    read_data_file.write(read_data[i] + '\n')
    
write_data_file.close()
read_data_file.close()
colors_file.close()


