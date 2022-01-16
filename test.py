import tensorflow as tf

test_list = [x+1 for x in range(100)]
ds = tf.data.Dataset.from_tensor_slices(test_list)
ds = ds.shuffle(32).repeat(2)
for elem in ds:
    print(elem.numpy())

# if __name__ == '__main__':
#     print("done")
