import string
all_letter = string.ascii_lowercase + string.digits
mapping = {}
mapping_inv = {}
i = 0
for x in all_letter:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1


num_chart = len(mapping)
print(num_chart)
# print(mapping)
print(mapping_inv)
