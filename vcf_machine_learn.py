#machine learn
import sparse, pysam, ast, glob, pickle, allel, random
import numpy as np
import configparser, argparse, sys, os
from random import sample
import pandas as pd
from Bio import SeqIO
import multiprocessing as mp
from itertools import repeat

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.metrics import BinaryAccuracy, AUC, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
import CLR
import gc, time, json

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

def get_model(input_shape, hp):
    model = Sequential()
    # Convolutional Layers
    #print("Add Conv")
    if hp["subset"] > 0:
        hp["size"][0] = (100,hp["subset"]*2)
        hp["stride"][0] = (5,hp["subset"]*2)
    for i in range(len(hp["filters"])):
        model.add(Conv2D(filters = hp["filters"][i][0], kernel_size = hp["size"][i], strides = hp["stride"][i],
                      activation='relu', kernel_regularizer = l2(l=hp["l2_reg"]),
                      input_shape=input_shape))
        model.add(Dropout(rate = hp["filters"][i][1]))
        if i == 1:
            model.add(AveragePooling2D(pool_size=(hp["max_pooling_size"],1), strides=hp["max_pooling_stride"]))
        if i in (3,5):
            model.add(MaxPooling2D(pool_size=(hp["max_pooling_size"],1), strides=hp["max_pooling_stride"]))
    # Pool
    #print("Add pool")
    #model.add(MaxPooling2D(pool_size=(hp["max_pooling_size"],1), strides=hp["max_pooling_stride"]))
    model.add(Flatten())
    # Linear Layers
    #print("Add dense layer")
    model.add(Dense(units = hp["dense_filters"], activation = 'relu',
                      kernel_regularizer = l2(l=hp["l2_reg"])
                      )
    )
    model.add(Dropout(rate = hp["dropout"]))

    # output layer
    #print("Add output")
    model.add(Dense(units = 1, activation = 'sigmoid',
                  kernel_regularizer = l2(l=hp["l2_reg"])))
    #myoptimizer = SGD(learning_rate=hp["base_lr"], momentum=hp["max_m"])
    myoptimizer = Adam(learning_rate=hp["base_lr"])
    model.compile(optimizer=myoptimizer,
                  loss="binary_crossentropy",
                  metrics=[
                          TruePositives(name='TP'),FalsePositives(name='FP'),
                          TrueNegatives(name='TN'), FalseNegatives(name='FN'),
                          BinaryAccuracy(name='acc50', threshold=0.5), BinaryAccuracy(name='acc90', threshold=0.9),
                          AUC(name='auroc', curve='ROC'), AUC(name='auprc', curve='PR')
                          ]
                  )
    model.summary()
    return model

def train_model_clr(x_train, y_train, hp):
    total = y_train.shape[0]
    weight_for_0 = (1 / np.sum(y_train==0))*(total)/2.0
    weight_for_1 = (1 / np.sum(y_train==1))*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    iterPerEpoch = (y_train.shape[0] - 200) / hp["batch"] 
    iterations = list(range(0,round((iterPerEpoch*hp["epoch"])+1)))
    step_size = (len(iterations)/(hp["n_cycles"]))/2
    
    # set cyclic learning rate
    scheduler = CLR.CyclicLR(base_lr=hp["base_lr"],
                                max_lr=hp["max_lr"],
                                step_size=int(step_size),
                                mode="triangular2")
    
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    #with strategy.scope():
        #model = get_model(x_train.shape[1:], hp)
        
    #5 kfold cross validation
    #splits data into 800 training and 200 test dataset
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    hists = list()
    for train, test in kfold.split(x_train, y_train):
        print(f'Training for fold {fold_no} ...')
        model = get_model(x_train.shape[1:], hp)
        hist = model.fit(x_train[train],
                        y_train[train],
                        batch_size = hp["batch"],
                        epochs = hp["epoch"],
                        verbose = 2,
                        class_weight = class_weight,
                        validation_data=(x_train[test], y_train[test]),
                        callbacks = [scheduler, 
                            LRTensorBoard(log_dir=os.path.join(hp["working_folder"],hp["model_name"]))])
        with open(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_"+str(fold_no))+"_hist.txt", 'w') as f:
            hist_df = pd.DataFrame(hist.history)
            hist_df.to_csv(f)
            f.close()
        hists.append(hist)
        fold_no += 1
        K.clear_session()
        gc.collect()
        scheduler._reset()
        time.sleep(30)
    return hists

def parse_vcf(vcfs, bed, seq_length, nsamples):
    vcf_reader = vcfpy.Reader(open(vcfs, 'r'))
    vcf_CHROM = []
    vcf_POS = []
    vcf_GT = []
    r_itr = 0
    for record in vcf_reader:
        sampling = 0
        idx = 0
        vcf_CHROM.append(record.CHROM) 
        vcf_POS.append(record.POS-1)
        vcf_GT.append([])
        for calls in record.calls:
            if sampling < nsamples:
                try:
                    haps = calls.data["GT"].replace("/", "|").split("|")
                    for hap in haps:
                        if hap == ".":
                            pass
                        elif int(hap) > 0: #only get haps that are not 0
                            vcf_GT[r_itr].append(idx)
                        idx += 1
                    sampling += 1
                except:
                    pass
        r_itr += 1

    vcf_pd = pd.DataFrame({'CHROM': vcf_CHROM, 'POS': vcf_POS, 'GT': vcf_GT})
    vcf_bed = pd.read_table(bed, header=None)
    
    x = []
    for idx in range(len(vcf_bed)):
        vcf_sub = vcf_pd[(vcf_pd["CHROM"] == vcf_bed.loc[idx][0]) & (vcf_pd["POS"] >= vcf_bed.loc[idx][1]) & (vcf_pd["POS"] < vcf_bed.loc[idx][2])].reset_index()
        vcf_array = np.tile(np.expand_dims(np.zeros(seq_length), axis=1), nsamples * 2)
        vcf_array = np.expand_dims(vcf_array, axis=2)
        for idx2 in range(len(vcf_sub)):
            vcf_array[(vcf_sub.loc[idx2][2] - vcf_bed.loc[idx][1]),  vcf_sub.loc[idx2][3]] = 1
        x.append(vcf_array)
    x = np.asarray(x).astype(np.float32)
    return x

def vcf2bin(vcfs, seq_length, nsamples):
    vcf_array = np.zeros(shape=(seq_length, nsamples * 2)).astype(np.int8)
    #vcf_array = np.expand_dims(vcf_array, axis=2)
    vcf_reader = pysam.VariantFile(vcfs)
    samples = list((vcf_reader.header.samples))
    for rec in vcf_reader.fetch():
        idx = 0
        for sample in samples:
            haps = rec.samples[sample]['GT']
            for hap in haps:
                if hap == ".":
                    pass
                elif int(hap) > 0: #only get haps that are not 0
                    vcf_array[int(rec.pos)-1,idx] = 1
            idx += 1
    vcf_reader.close()
    #sorting the entire matrix with invariable sites takes a long time
    #can just get sorting order from variant only matrix and use that to sort big matrix
    vcf_allel = allel.read_vcf(vcfs, fields="calldata/GT")
    b = np.reshape(vcf_allel['calldata/GT'], (vcf_allel['calldata/GT'].shape[0], nsamples * 2))
    c = np.swapaxes(b, 0, 1)
    vcf_array = np.swapaxes(vcf_array, 0, 1)[sort_min_diff(c)]
    vcf_array = np.swapaxes(vcf_array, 0, 1)
    vcf_array = np.expand_dims(vcf_array, axis=2)
    return vcf_array

def sort_min_diff(matrix):
    #this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    #this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    #assumes your input matrix is a numpy array (Flagel et al. 2019 MPE)  
    mb = NearestNeighbors(n_neighbors=len(matrix), metric='manhattan').fit(matrix)
    v = mb.kneighbors(matrix)
    smallest = np.argmin(v[0].sum(axis=1))
    return v[1][smallest]

def encode_sequence(vcf_folder, seq_length, nsamples, pos="+"):
    # encode and save
    vcf_list = glob.glob(vcf_folder+"/*.vcf")
    # run in parallel
    pool = mp.Pool(mp.cpu_count()-2)
    x = pool.starmap(vcf2bin, zip(vcf_list, repeat(seq_length), repeat(nsamples)))
    x = np.asarray(x).astype(np.int8)
    if pos == "+":
        y = np.ones(len(x))
    elif pos == "-":
        y = np.zeros(len(x))
    y = np.asarray(y).astype(np.int8)
    print(x.shape, y.shape)
    return x, y
    
def load_data(hp):
    if os.path.exists(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_train_fasta.npz")) and not hp["rewrite"]:
        print("output folder model_output already contains loaded data, loading that (to recode use --force)")
        print("loading training data...")
        x_train = sparse.load_npz(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_train_fasta.npz")).todense()
        y_train = sparse.load_npz(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_train_labels.npz")).todense()
        print(f'There are {np.count_nonzero(y_train == 1)} positives and {np.count_nonzero(y_train == 0)} negatives.')
    else:
        # encode and save
        print("loading training data...")
        (x1_train, y1_train) = encode_sequence(hp["pos_folder"], hp["length"], hp["nsamples"])
        (x2_train, y2_train) = encode_sequence(hp["neg_folder"], hp["length"], hp["nsamples"], pos="-")
        x_train = np.concatenate((x1_train, x2_train))
        y_train = np.concatenate((y1_train, y2_train))
        print(f'There are {x1_train.shape[0]} positives and {x2_train.shape[0]} negatives.')
        sparse.save_npz(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_train_fasta.npz"), sparse.COO(x_train), compressed=True)
        sparse.save_npz(os.path.join(hp["working_folder"],hp["model_name"], hp["model_name"]+"_train_labels.npz"), sparse.COO(y_train), compressed=True)
    return x_train, y_train
    
def main(hp):
    if hp["train"]:
        # load data
        x_train, y_train = load_data(hp)
        if hp["subset"] > 0:
            subset = int(x_train.shape[2]/2)
            randos = random.sample(range(0,subset-1,2), int(hp["subset"]/2))
            randos = sorted(randos + [x+1 for x in randos] + [x+subset for x in randos] + [x+subset+1 for x in randos])
            x_train = x_train[:,:,randos,:]
            hp["model_name"] = hp["model_name"]+"_s"+str(subset)
            print(x_train.shape)
            sum_values = np.sum(x_train, axis=(2,3))
            average_values = np.mean(sum_values, axis=0)
            indexed_average_values = np.column_stack((np.arange(len(average_values)), average_values))
            np.savetxt('average_values.txt', indexed_average_values, fmt='%d %f', header='Index Average', comments='')
            #if not os.path.exists(os.path.join(hp["working_folder"],hp["model_name"])):
            #    os.makedirs(os.path.join(hp["working_folder"],hp["model_name"]))
        #models_all = train_model_clr(x_train, y_train, hp)
        #with open(os.path.join(hp["working_folder"],hp["model_name"],hp["model_name"]+".pkl"), 'wb') as pickle_file:
        #    pickle.dump(models_all, pickle_file)
        #    pickle_file.close()
        
    if hp["eval"]:
        # Kfold is 10
        models_all = pickle.load(open(os.path.join(hp["working_folder"],hp["model_name"],hp["model_name"]+".pkl"), 'rb'))
        for each_k in models_all:
            loaded_model = each_k.model
            for key in hp["eval_vcf"].keys():
                print("Evaluating ", key)
                evals = parse_vcf(hp["eval_vcf"][key], hp["eval_bed"][key], hp["length"], hp["nsamples"])
                print(len(evals))
                scores = loaded_model.predict(evals).ravel()
        with open(os.path.join(hp["working_folder"],"model_output",hp["model_name"]+"_"+key+"_prob.txt"), "w") as prob_file:
            vcf_bed = pd.read_table(hp["eval_bed"][key], header=None)
            for idx in range(len(vcf_bed)):
                prob_file.write('{}\t{}\n'.format(vcf_bed.loc[idx][3],scores[idx]))
        prob_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Config File")
    parser.add_argument("--train", action='store_true', help="Train Model")
    parser.add_argument("--eval", action='store_true', help="Evaluate Model")
    parser.add_argument("--force", action='store_true', help="Force Rewrite")
    parser.add_argument("--pos_folder", type=str, help="Folder of Vcfs of Class 1")
    parser.add_argument("--neg_folder", type=str, help="Folder of Vcfs of Class 2")
    parser.add_argument("--outname", type=str, help="output model name")
    parser.add_argument("--subsample", type=int, help="subsample from main data", default=0)

    options, args = parser.parse_known_args()
    
    config = configparser.ConfigParser()
    config.read(options.config)
    
    hp = {"base_lr" : float(config["scheduler"]["base_lr"]), "max_lr" : float(config["scheduler"]["max_lr"]), 
            "l2_reg" : float(config["scheduler"]["l2_reg"]), "n_cycles" : float(config["scheduler"]["n_cycles"]),
            "filters" : ast.literal_eval(config["model"]["2D_conv_filters"]), 
            "size" : ast.literal_eval(config["model"]["2D_conv_size"]), "stride" : ast.literal_eval(config["model"]["2D_conv_stride"]),
            "max_pooling_size" : int(config["model"]["max_pooling_size"]), "max_pooling_stride" : int(config["model"]["max_pooling_stride"]),
            "dense_filters" : int(config["model"]["connected_dense_filters"]), "dropout" : float(config["model"]["connected_dense_dropout"]), 
            "train" : False, "eval" : False, 
            "rewrite" : options.force,
            "working_folder" : config["options"]["working_folder"]}
    
    if not options.train and not options.eval:
        print("You did not choose --train or --eval, will do both")
        options.train = True
        options.eval = True  

    if options.train:
        hp["train"]=True
        if options.pos_folder == None:
            hp["pos_folder"] = config["data"]["pos_folder"]
        else:
            hp["pos_folder"] = options.pos_folder
        if options.neg_folder == None:
            hp["neg_folder"] = config["data"]["neg_folder"]
        else:
            hp["neg_folder"] = options.neg_folder
        hp["batch"] = int(config["options"]["batch"])
        hp["epoch"] = int(config["options"]["epoch"])
        hp["length"] = int(config["options"]["max_seq_len"])
        hp["nsamples"] = int(config["options"]["samples"])
        hp["subset"] = options.subsample
        if options.outname == None:
            hp["model_name"] = config["options"]["name"]
        else:
            hp["model_name"] = options.outname
        if not os.path.exists(os.path.join(hp["working_folder"],hp["model_name"])):
            os.makedirs(os.path.join(hp["working_folder"],hp["model_name"]))
            
    if options.eval:
        hp["eval"] = True
        hp["length"] = int(config["options"]["max_seq_len"])
        hp["nsamples"] = int(config["options"]["samples"])
        hp["working_folder"] = config["options"]["working_folder"]
        hp["eval_vcf"] = {}
        for x in config["eval_vcf"]:
            hp["eval_vcf"][x] = config["eval_vcf"][x]
        hp["eval_bed"] = {}
        for x in config["eval_bed"]:
            hp["eval_bed"][x] = config["eval_bed"][x]
        hp["model_name"] = config["options"]["name"]

    main(hp)
