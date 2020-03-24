import numpy as np
import torch
import torch.nn as nn

# Define model
NUM_DIGITS = 10
NUM_HIDDEN = 100
BATCH_SIZE = 128
model = nn.Sequential(
    nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    nn.ReLU(),
    nn.Linear(NUM_HIDDEN, 4),  # logits, after softmax, get a probablity distribution
).cuda()
# 拟合两种分布的相似度有多高, 希望两种分布越接近越好
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))

def binary_encode(i, num_digits):
    # i每次向右移d位，然后&1取最后一位
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


# Train phase
# torch.Tensor(ndarray),创建tensor,注意torch.Tensor([ndarray])
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
for epoch in range(10000):
    # the third parameter in range is the step
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        print("Epoch:", epoch, "Loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test phase
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
if torch.cuda.is_available():
    testX = testX.cuda()
# test without remember gradient, in case of taking my memory
with torch.no_grad():
    testY = model(testX)
    # 1 respresents take the max in every row(0 respresents column）
    # testY.max(1) will return two matrices,[0] is the max number,[1] is the index of the max number
    # zip the number in range and list into a tuple
    prediction = zip(range(1, 101), list(testY.max(1)[1].data))
    print([fizz_buzz_decode(i, x) for i, x in prediction])


