import keras as k
from tensorflow.keras import backend as K
from keras.layers import Input, Dense, Concatenate, concatenate, Flatten, Dropout
from keras.models import Model
from keras.losses import cosine_proximity as cosine
import numpy as np
from scipy.spatial import distance
import math
import random

# mush load these two package for loading data
import msgpack
import msgpack_numpy as m
m.patch()

import pickle

dim = 8000 # Size of MS vector
latent_dim = 256 # Size of encoding space
intermediate_dim = 1024 # Size of dense layers
epsilon_std = 1.0
# Under the hood code, to process MS/MS information from original project.
y_type = 'float32'
x_type = 'float32'

noice = 0 #1e-8

max_it = 1.0e4
min_it = 0.0
thres = 0.1
it_scale = max_it + min_it
raw_thres = max_it / 1.0e4

precision = 0.2 # bin width
low = 180.0
dim = 8000
upper = math.floor(low + dim * precision)
mz_scale = 2000.0
max_mz = 1500

max_out = dim

it_scale = max_it

max_peaks = 400
topall = 150
topk = 10
max_len = 22
max_in = max_len + 2
max_charge=4
oh_dim = 28

def pre(): return precision
def get_pre(): return precision
getm = lambda : mode
getlow = lambda : low

def mz2pos(mz, pre=pre()): return int(round((mz - low) / pre))
def pos2mz(pos, pre=pre()): return pos * pre + low

def asnp32(x): return np.asarray(x, dtype='float32')

def np32(x): return np.array(x, dtype='float32')

def zero32(shape): return np.zeros(shape, dtype='float32')

def cos(x, y): return 1 - distance.cosine(x, y)

def scale(v, _max_it = 1):
    c0 = np.max(v)
    if c0 == _max_it or c0 == 0: return v #no need to scale     
    c = _max_it / c0
    return v * c

def normalize(it):    
    it[it < 0] = 0
    return np.sqrt(np.sqrt(it))

def encode(seq, out=None, l=1): #with smoothing
    if out is None:
        em = np.zeros((max_in, oh_dim), dtype='float32')
    else:
        em = out
    
    for i in range(len(seq)):
        em[i][seq[i]] = l
    
    em[len(seq)][27] = 1 # end char, no smooth
    
    for i in range(len(seq) + 1, max_in): em[i][0] = 1 # padding
        
    return em

def toseq(pep):
    pep = pep.replace('I', 'L')
    return [ord(char) - 64 for char in pep]

def onehot(pep, out=None, **kws):
    return encode(toseq(pep), out=out, **kws)

def flat(mz, it, precision = pre(), dim=dim, out=None, low=None):
    if low is None: low = getlow()
    v = np.zeros(dim, dtype = y_type) if out is None else out

    length = len(v)

    xs = (mz - low) / precision
    xs = xs.astype('int32')

    for i, pos in enumerate(xs):
        if pos < 0 or pos >= length: continue
        v[pos] = max(v[pos], it[i])
        
    v[0] = 0 #clear xs at 0
    return v

def vectorlize(sp, precision, out=None, **kws):
    mz, it, mass = sp['mz'], sp['it'], sp['mass']
    
    it = scale(it)
    it = normalize(it)
    
    return flat(mz, it, precision=precision, out=out, **kws)

def loadmp(fn):
    f = open(fn,'rb')
    db = msgpack.load(f, use_list=False, raw=False)
    f.close()
    return db

def load(fn): return loadmp(fn + "-np.mp")

def preproc(sps):
    sps = [x for x in filter(lambda sp: len(sp['pep']) <= max_len, sps)]

    peps = []
    mzs = []
    x = np.full((len(sps), max_in, oh_dim), noice, dtype=x_type)
    y = np.full((len(sps), dim), noice, dtype=y_type)

    for i in range(len(sps)):
        sp = sps[i]

        peps.append(sp['pep'])
        mzs.append(sp['mass'])

        onehot(sp['pep'], out=x[i])
        vectorlize(sp, precision=precision, out=y[i])

    return sps, x, y
def get_sps(sps):
    sps = [x for x in filter(lambda sp: len(sp['pep']) <= max_len, sps)]
    return sps
def preproc_all():
    sps = []
    sps += get_sps(load('kall')['2'])
    sps += get_sps(load('krust')['2'])
    sps += get_sps(load('hcd')['2'])
    sps += get_sps(load('ham')['2'])
    peps = []
    mzs = []
    x = np.full((len(sps), max_in, oh_dim), noice, dtype=x_type)
    y = np.full((len(sps), dim), noice, dtype=y_type)

    for i in range(len(sps)):
        sp = sps[i]

        peps.append(sp['pep'])
        mzs.append(sp['mass'])

        onehot(sp['pep'], out=x[i])
        vectorlize(sp, precision=precision, out=y[i])

    return x, y

x = Input(shape=(dim,), name='spectrum') # input
cond = Input(shape = (max_in, oh_dim,), name='pep_OHE')
cond_flat = Flatten(name = 'pep')(cond)
inputs = Concatenate(name='inputs')([x, cond_flat])

# calculate hidden variable, change to suitable structure !!!!
h = Dense(intermediate_dim, activation='relu', name='dense1')(inputs)
h = Dense(intermediate_dim, activation='relu', name='dense2')(h)

# get mean and var of p(Z|X)
z_mean = Dense(latent_dim, name='mean')(h)
z_log_var = Dense(latent_dim, name='std')(h)

# reparameter skill
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# do x' = z * var + mean
z = k.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
z_cond = Concatenate()([z, cond_flat])
# the decoder, modify it to suitable structure !!!!
decoder_h = Dense(intermediate_dim, activation='relu', name = "decoder")
decoder_mean = Dense(dim, activation='sigmoid', name='output')

h_decoded = decoder_h(z_cond)
h_decoded = Dense(intermediate_dim, activation='relu', name = "dense10")(h_decoded)
h_decoded = Dense(intermediate_dim, activation='relu', name = "dense20")(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(inputs = [x,cond], outputs = x_decoded_mean)


xcos_loss = cosine(x, x_decoded_mean) # Cosine loss to compare MS/MS vectors
k1_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(10*xcos_loss + k1_loss) # Weighted loss

vae.add_loss(vae_loss)
vae.compile(optimizer=k.optimizers.Adam(0.003))
vae.summary()
# then do the training, note that we don't realy need a output accualy
# vae.fit(y, epochs=10, batch_size=128, validation_data=(y, None))
x, y = preproc_all() # Load data
vae.fit([y,x], epochs = 15, batch_size = 64, shuffle = True) # Train

vae.save('full_vae.h5')
