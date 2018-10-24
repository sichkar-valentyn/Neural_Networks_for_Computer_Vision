# Here we're getting data-sets CIFAR-10
# To run this file open terminal and move to this directory 'cd datasets'
# Run this file './get_CIFAR-10.sh'
# If there is error that 'permission denied' change permission by following command
# 'sudo chmod +x get_CIFAR-10.sh'
# And run again './get_CIFAR-10.sh'


# Downloading archive from official resource
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Unzipping archive
tar -xzvf cifar-10-python.tar.gz

# Deleting non-needed anymore archive
rm cifar-10-python.tar.gz


# As a result there has to appear new folder 'cifar-10-batches-py' with following files
# data_batch_1
# data_batch_2
# data_batch_3
# data_batch_4
# data_batch_5
# batches.meta
# test_batch
# readme.html
