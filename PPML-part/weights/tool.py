items = []

filename = "model_layer3_block0_bias3.txt"
with open(filename, "r") as f:
    lines = f.readlines()

for line in lines:
    if not line in items:
        items.append(line)

print(items)

new_items = [item * 16 * 16 for item in items]
print(new_items)


with open(filename, "w") as f:
    f.writelines(new_items)
