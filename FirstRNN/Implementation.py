import math
import numpy as np

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# this generates a sin wave with length 200, which is what np.arange's first parameter here means.
# shape is (200,)
sin_wave = np.array([math.sin(x) for x in np.arange(200)])

# arrays holding our data
X = []
Y = []

seq_len = 50
# we get 150 num_records
num_records = len(sin_wave) - seq_len

# we do this 100 times. For each set we take the next 50 values.
# starts with : [0:50], [1:51], [2:52], ... [99:149], of course excluding the last value.
# the last value (index 50 for first subset) is put in Y, because that is the value we want to predict.
for i in range(num_records - 50):
    X.append(sin_wave[i:i + seq_len])
    Y.append(sin_wave[i + seq_len])

X = np.array(X)
# this function adds brackets at a certain level of the array.
# axis = 0 is for outer brackets. So it adds [ array ].
# axis = 1 is for some inner brackets. So for a 1D array, it adds brackets around all indexes in that 1D array. [ [array[0]], [array[1]],...]
# axis = 2 is for some deeper inner brackets. It needs to be at least a 2D array, so that it puts it on each of the indexes of that 2D array.
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)


# making the validation sets. Same thing as before, using the next possible 50 sets we can create out of the sin wave.
X_val = []
Y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i + seq_len])
    Y_val.append(sin_wave[i + seq_len])

X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)


# learning rate of the algorithm
learning_rate = 0.0001
# nb of epochs for the machine to learn
nepoch = 25
# length of sequence
T = 50
# dimension of the hidden layer
hidden_dim = 100
# dimension of the output layer
output_dim = 1

# back propagation through time value
bptt_truncate = 5
# using gradient value clipping: if a gradient below -10, it gets the min_clip_value. If a gradient above 10, it gets
# the max_clip_value.
min_clip_value = -10
max_clip_value = 10

# different weights layers
# first takes your T dimension x hidden_dim
U = np.random.uniform(0, 1, (hidden_dim, T))
# hidden_dim x hidden_dim for the hidden layer
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
# output layer weights.
V = np.random.uniform(0, 1, (output_dim, hidden_dim))



# epochs for the machine to learn
for epoch in range(nepoch):

    # LOSS ON TRAINING DATA
    # check loss on train
    loss = 0.0

    # do a forward pass to get prediction
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]  # get input, output values of each record
        # here, prev-s is the value of the previous activation of hidden layer; which is initialized as all zeros
        prev_s = np.zeros((hidden_dim, 1))

        for t in range(T):
            # this is my sequence. I am going to take a single value of the sequence.
            new_input = np.zeros(x.shape)  # we then do a forward pass for every timestep in the sequence
            new_input[t] = x[t]  # for this, we define a single input for that timestep

            # now we multiply our value by our weights of the first level
            mulu = np.dot(U, new_input)
            # we then multiply our weights of the second level by the previous activation of the hidden layer.
            mulw = np.dot(W, prev_s)

            # we add those two matrices
            add = mulw + mulu
            # and we activate that result.
            s = sigmoid(add)
            # we then process the result with our weights of the level V.
            mulv = np.dot(V, s)
            # so this is the activation result of the mulu + mulw result.
            prev_s = s

        # calculate error (attributing value to error with loss) of mulv, calculated in the for loop.
        loss_per_record = (y - mulv) ** 2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])

    # LOSS ON VALIDATION DATA
    # (Same thing as was done on training data)
    # check loss on val
    val_loss = 0.0
    for i in range(Y_val.shape[0]):
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s

        loss_per_record = (y - mulv)**2 / 2
        val_loss += loss_per_record
    val_loss = val_loss / float(y.shape[0])

    print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)


    # TRAINING THE DATA - Still inside epoch loop.
    # train model
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]

        layers = []
        prev_s = np.zeros((hidden_dim, 1))
        # derivatives of the weights (for backprop)
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        dW = np.zeros(W.shape)

        # derivatives of the weights through time?? bptt
        dU_t = np.zeros(U.shape)
        dV_t = np.zeros(V.shape)
        dW_t = np.zeros(W.shape)

        # see below. dU_i and dW_i reproduce the matrix as it was in the forward pass as mulv and mulu using the input at
        # timestep T.
        dU_i = np.zeros(U.shape)
        dW_i = np.zeros(W.shape)

        # forward pass
        for t in range(T):
            # similar to what is commented above.
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            # adds an index value 's' and 'prev_s'
            layers.append({'s': s, 'prev_s': prev_s})
            prev_s = s

        # BACK PROPAGATION THROUGH TIME
        # derivative of pred
        dmulv = (mulv - y)

        # backward pass
        for t in range(T):
            # REMEMBER : With matrix multiplications, AB != BA
            # taking the activation results and multiply them by the difference between our answer and the actual answer
            dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
            # multiply our weights by the difference between our answer and the actual answer (gradient)
            # this returns us to the sigmoid step.
            # Antiderivative of V level
            dsv = np.dot(np.transpose(V), dmulv)

            ds = dsv
            # the antiderivative of sigmoid is result*(1-result).
            # add is the addition of the last timestep: mulw+mulu. This is because it has memory of all timesteps.
            # why is this the antiderivative of sigmoid using add and multiplied by the sigmoid antiderivative value??
            dadd = add * (1 - add) * ds
            # ones_like returns matrix with only 1s. We put this gradient on all values of the dmulw matrix.
            dmulw = dadd * np.ones_like(mulw)
            # for some reason, only apply the gradient matrix to the prev_s.
            # Antiderivative of W level
            dprev_s = np.dot(np.transpose(W), dmulw)

            # the reason we call this back propagation bptt is because the antiderivative of the values depend on
            # multiple variables from earlier time steps because of prev_s. prev_s is also what makes the RNN keep
            # some memory.

            # this is where bptt comes into play
            for i in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
                # Addition of antiderivative of V + W
                ds = dsv + dprev_s
                # add is always same. Why redo it here?
                # dadd = add * (1 - add) * ds

                # Replicating again what is done outside the loop... except for dmulu
                # dmulw = dadd * np.ones_like(mulw)
                dmulu = dadd * np.ones_like(mulu)

                # dW_i is the resulting matrix from the activation results at a certain time step multiplied by W weights
                dW_i = np.dot(W, layers[t]['prev_s'])
                # Same thing as above...
                # dprev_s = np.dot(np.transpose(W), dmulw)

                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                # dU_i is the resulting matrix from the U weights multiplied by an actual input value (like in forward
                # pass)
                dU_i = np.dot(U, new_input)
                # now this is the antiderivative of the U layer. notice dmulu is set using gradients.
                dx = np.dot(np.transpose(U), dmulu)

                # this is adding the inputs together (why??).
                dU_t += dU_i
                dW_t += dW_i

            # this is adding the gradients together.
            dV += dV_t
            # this is adding the inputs together (why??)
            dU += dU_t
            dW += dW_t

            # Gradient value clipping applied.
            # Notice boolean indexing. This makes all values above a certain threshold equal the threshold you set.
            if dU.max() > max_clip_value:
                dU[dU > max_clip_value] = max_clip_value
            if dV.max() > max_clip_value:
                dV[dV > max_clip_value] = max_clip_value
            if dW.max() > max_clip_value:
                dW[dW > max_clip_value] = max_clip_value

            if dU.min() < min_clip_value:
                dU[dU < min_clip_value] = min_clip_value
            if dV.min() < min_clip_value:
                dV[dV < min_clip_value] = min_clip_value
            if dW.min() < min_clip_value:
                dW[dW < min_clip_value] = min_clip_value

        # update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW