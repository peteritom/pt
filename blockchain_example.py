# A simple, basic blockchain implementation
# followed tutorial from NeuralNine: https://www.youtube.com/watch?v=pYasYyjByKI
# created by PT
# date: 30/07/2021

import hashlib

class PTCoinBlock:

    def __init__(self, prev_block_hash, trans_list):
        self.prev_block_hash = prev_block_hash
        self.trans_list = trans_list

        self.block_data = "-".join(trans_list) + "-" + prev_block_hash
        self.block_hash = hashlib.sha256(self.block_data.encode()).hexdigest()

if __name__ == '__main__':

    trans1 = "Tom sends 10 PTC to Marc"
    trans2 = "Marc sends 0.01 PTC to Adam"
    trans3 = "Adam sends 4 PTC to Tom"
    trans4 = "Adam sends 13.2 PTC to Kate"
    trans5 = "Tom sends 5 PTC to Adam"
    trans6 = "Kate sends 101 PTC to Tom"

    # use case #1 - initialize block chain

    initial_block = PTCoinBlock("Init String", [trans1, trans2])

    print (initial_block.block_data)
    print (initial_block.block_hash)

    # use case #2 - initialize block chain with slightly different data

    trans2_d = "Marc sends 0.009 PTC to Adam"

    initial_block_d = PTCoinBlock("Init String", [trans1, trans2_d])

    print (initial_block_d.block_data)
    print (initial_block_d.block_hash)

    # use case #3 - create a second and third block

    second_block = PTCoinBlock(initial_block.block_hash, [trans3, trans4])
    third_block = PTCoinBlock(second_block.block_hash, [trans5, trans6])

    print (second_block.block_data)
    print (second_block.block_hash)
    print (third_block.block_data)
    print (third_block.block_hash)

    # use case #4 - check the integrity of the blockchain, alter the initial block slightly

    initial_block = PTCoinBlock("Init String", [trans1, trans2_d])
    second_block = PTCoinBlock(initial_block.block_hash, [trans3, trans4])
    third_block = PTCoinBlock(second_block.block_hash, [trans5, trans6])

    print("\n ALTERED BLOCKCHAIN \n")
    print (initial_block.block_data)
    print (initial_block.block_hash)
    print (second_block.block_data)
    print (second_block.block_hash)
    print (third_block.block_data)
    print (third_block.block_hash)
