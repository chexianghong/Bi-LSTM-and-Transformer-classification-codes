import os
from gio import geo_raster as ge
from gio import geo_raster_ex as gx
from gio import geo_base as gb
import numpy as np
import datetime
import math
import re
import tensorflow as tf
import pandas as pd
from lib_wetland import calculate_index
from lib_wetland import linear_regress
from lib_wetland import cal_metrics
from lib_wetland import machine_learning
from lib_wetland import LSTM_model
import time
import pickle
import logging
from sklearn.model_selection import train_test_split
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config = config)

bnd_ids = ['Aerosol', 'Blue', 'Green','Red', 'NIR', 'SWIR1', 'SWIR2','NDVI', 'NDMI','NDWI','NDIIB7','EVI','SAVI','MSAVI']
EPOCH = 80
batch_size = 32

def read_train_samples(f_csv):
    pt_x = {}
    pt_y = {}
    with open(f_csv, 'r') as _f_in:
        for line in _f_in.read().splitlines():
            _l = line.split(',')
            if _l[0] == 'ID':
                continue

            _id = int(_l[0])
            _ts = int(_l[1])

            pt_x[_id] = pt_x.get(_id, {})
            pt_x[_id][_ts] = [float(_val) for _val in _l[2:-1]]
            pt_y[_id] = int(_l[-1])

    ls_x = []
    ls_y = []
    for i in list(pt_x.keys()):
        _ts = pt_x[i].keys()
        _ts.sort()
        ls_x.append(np.array([pt_x[i][j] for j in _ts]))
        ls_y.append(pt_y[i])

    return ls_x, ls_y

def output_samples(_x, _y, _ids, f_out):
    ls = ['doy,timestep,'+','.join(bnd_ids)+',LC']
    for i in range(len(_y)):
        for j in range(_x[0].shape[0]):
            _doy = (_edge + j + 1) * _days - _days/2
            ls.append('%s,%s,%s,%s,%s' % (_ids[i], _doy, j, ','.join([str(_v) for _v in _x[i][j,:]]),_y[i]))

    with open(f_out, 'w') as _fo:
        _fo.write('\n'.join(ls))


def normlize(_x, x_mean, x_std):
    _x = np.ma.masked_equal(np.array(_x), -9999).astype(np.float32)
    _x = (_x - x_mean) / x_std
    _x = _x.filled(-9999)
    return _x

def sampling(ls_x, ls_y, _idx, dd_norm, _ty):
    ls_x_temp = [_x[0] for _x in ls_x]
    _x = np.ma.masked_equal(np.array(ls_x_temp), -9999).astype(np.float32)
    x_mean = np.array([_x[:, :, i].mean() for i in range(len(bnd_ids))]).astype(np.float32)
    x_std = np.array([_x[:, :, i].std() for i in range(len(bnd_ids))]).astype(np.float32)

    lt = [','.join(str(_val) for _val in list(x_mean)), ','.join(str(_val) for _val in list(x_std))]
    f_norm = os.path.join(dd_norm, 'norm_para_%s.csv' % _ty)
    with open(f_norm, 'w') as _fo:
        _fo.write('\n'.join(lt))

    ps = {}
    _rios = [4/5, 3/4, 2/3, 1/2]
    for i in range(4):
        x1, x2, y1, y2 = train_test_split(ls_x, ls_y, test_size = _rios[i], random_state = 1, stratify = ls_y)
        ps[i] = [x1,y1]
        ls_x = x2
        ls_y = y2
    ps[4] = [x2, y2]
    
    y_test = ps[_idx][1]
    x_test = normlize([_v[0] for _v in ps[_idx][0]], x_mean, x_std)
    x_test_id = [_v[1] for _v in ps[_idx][0]]
    
    _ids = list(ps.keys())
    _ids.sort()
    x_train = []
    y_train = []
    for _id in _ids:
        if _id != _idx:
            x_train.extend([_v[0] for _v in ps[_id][0]])
            y_train.extend(ps[_id][1])
            
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.04, random_state=1, stratify=y_train)
    x_train = normlize(x_train, x_mean, x_std)
    x_validation = normlize(x_validation, x_mean, x_std)
    return  x_train, x_test, x_validation,y_train, y_test,y_validation, x_mean, x_std, x_test_id

def filter_invalid(x, y):
    _xx = []
    _yy = []
    _num = x.shape[0]
    for i in range(_num):
        if np.sum(x[i,:,0]==-9999) == 14:
            continue
        _xx.append(x[i,:,:])
        _yy.append(y[i])
    return np.array(_xx), _yy

def parse_file(ls_agg_x, ls_interp_x, ls_y, _ff, n_obs):
    with open(_ff, 'r') as _f_in:
        _info = _f_in.read().splitlines()
        for i in range(1, len(_info), n_obs):
            _info_s = _info[i].split(',')
            _ty = int(_info_s[-1])
            if _ty in [66, 208, 211, 49, 111, 71, 217, 64, 152, 205]:
                continue
            ls_x_tmp = []
            for line in _info[i:i+n_obs]:
                _l = line.split(',')
                ls_x_tmp.append([float(_v) for _v in _l[:-5]])

            ls_agg_x.append((np.array(ls_x_tmp[:n_obs//2], dtype=np.int16), '%s,%s' % (_info_s[-3], _info_s[-2])))
            ls_interp_x.append((np.array(ls_x_tmp[n_obs//2:], dtype=np.int16), '%s,%s' % (_info_s[-3], _info_s[-2])))
            ls_y.append(int(_info_s[-1]))

def stat(y):
    ps = {}
    for _v in y:
        ps[_v+1] = ps.get(_v+1, 0)
        ps[_v + 1] = ps[_v + 1] +1
    return len(y), ps.items()

def output_norm(ls_x, dd_norm, _ty):
    ls_x_temp = [_x[0] for _x in ls_x]
    _x = np.ma.masked_equal(np.array(ls_x_temp), -9999).astype(np.float32)
    x_mean = np.array([_x[:, :, i].mean() for i in range(len(bnd_ids))]).astype(np.float32)
    x_std = np.array([_x[:, :, i].std() for i in range(len(bnd_ids))]).astype(np.float32)

    lt = [','.join(str(_val) for _val in list(x_mean)), ','.join(str(_val) for _val in list(x_std))]
    f_norm = os.path.join(dd_norm, 'norm_para_16d_%s.csv' % _ty)
    with open(f_norm, 'w') as _fo:
        _fo.write('\n'.join(lt))
    return x_mean, x_std

def filter_test_id(x, x_id):
    x_test_id = []
    _num = x.shape[0]
    for i in range(_num):
        if np.sum(x[i, :, 0] == -9999) == 14:
            continue
        x_test_id.append(x_id[i])
    return x_test_id

def label_crop(ls_y):
    pt_crop = {}
    _crop_ids = list(set(ls_y))
    for i in range(len(_crop_ids)):
        pt_crop[_crop_ids[i]] = i

    ls_yy_label = []
    for _y in ls_y:
        ls_yy_label.append(pt_crop[_y])
    return pt_crop, ls_yy_label

def _task():

    using_kt = False
    ff_sam = '/mnt/data_1/chexh/NSF_data/samples/samples_crop/train_samples_11SKA_2021.csv'
    dd_model = '/mnt/data_1/chexh/NSF_data/result_lstm_crop_f/model'
    dd_hist = '/mnt/data_1/chexh/NSF_data/result_lstm_crop_f/hist'
    dd_norm = '/mnt/data_1/chexh/NSF_data/result_lstm_crop_f/norm'
    dd_validation = '/mnt/data_1/chexh/NSF_data/result_lstm_crop_f/validation'

    os.path.exists(dd_model) or os.makedirs(dd_model)
    os.path.exists(dd_hist) or os.makedirs(dd_hist)
    os.path.exists(dd_norm) or os.makedirs(dd_norm)
    os.path.exists(dd_validation) or os.makedirs(dd_validation)

    ff_lc_id = os.path.join(dd_norm, 'lc_id_%s.csv')

    ls_agg_x = []
    ls_interp_x = []
    ls_y = []
    parse_file(ls_agg_x, ls_interp_x, ls_y, ff_sam, 28)

    pt_label, ls_y = label_crop(ls_y)
    pt_label = {v: k for k, v in pt_label.items()}

    # output the land cover ids (idx, code_id)
    lt_ids = ['%s,%s' % (_k, _v) for _k, _v in pt_label.items()]
    with open(ff_lc_id, 'w') as _fo:
        _fo.write('\n'.join(lt_ids))

    a = time.time()
    for _version in [1,2,3,4,5]:
        for _seed in range(5):

            _ty = '16d_%s_v%s' % (_seed, _version)
            dd_model_msk = os.path.join(dd_model, 'lr_mask_%s' % _ty)
            if os.path.exists(dd_model_msk):
               continue
            else:
                os.mkdir(dd_model_msk)
            dd_model_int = os.path.join(dd_model, 'lr_interp_%s' % _ty)
            os.path.exists(dd_model_int) or os.mkdir(dd_model_int)


            X_train, X_test, X_validation, ls_train_y, ls_test_y, ls_validation_y, x_mean, x_std, x_test_id = sampling(ls_agg_x,  ls_y, _seed, dd_norm, '16d_mask')
            train_n = X_train.shape[0]
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, np.array(ls_train_y))).shuffle(train_n + 1).batch(batch_size)
            validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, np.array(ls_validation_y))).batch(batch_size)
            del X_train, X_validation, ls_train_y, ls_validation_y
            LSTM_model.trainings_val_class2(len(lt_ids), train_n, batch_size, train_ds, validation_ds, dd_model_msk,os.path.join(dd_hist, 'lr_mask_%s.csv' % _ty), EPOCH, False, 0.0001,True)

            del train_ds, validation_ds

            X_train1, X_test1, X_validation1, ls_train_y1, ls_test_y1, ls_validation_y1, x_mean1, x_std1, x_test_id1 = sampling(ls_interp_x, ls_y, _seed, dd_norm, '16d_interp')
            train_n = X_train1.shape[0]
            train_ds1 = tf.data.Dataset.from_tensor_slices((X_train1, np.array(ls_train_y1))).shuffle(train_n + 1).batch(batch_size)
            validation_ds1 = tf.data.Dataset.from_tensor_slices((X_validation1, np.array(ls_validation_y1))).batch(batch_size)
            del X_train1, X_validation1, ls_train_y1, ls_validation_y1
            LSTM_model.trainings_val_class2(len(lt_ids),train_n, batch_size, train_ds1,validation_ds1, dd_model_int, os.path.join(dd_hist, 'lr_interp_%s.csv' % _ty), EPOCH, False, 0.0001, False)
            del train_ds1, validation_ds1

            _model_msk = tf.keras.models.load_model(dd_model_msk)
            X_test = X_test.astype(np.float32)
            _pred = _model_msk.predict(X_test, batch_size=5000)
            _pred = tf.argmax(_pred, 1)
            _pred = np.array(_pred).reshape((X_test.shape[0],))

            _model_int = tf.keras.models.load_model(dd_model_int)
            X_test1 = X_test1.astype(np.float32)
            _pred1 = _model_int.predict(X_test1, batch_size=5000)
            _pred1 = tf.argmax(_pred1, 1)
            _pred1 = np.array(_pred1).reshape((X_test1.shape[0],))

            ls = ['row,col,agg,agg_pred,int_pred,valid_obs']
            for i in range(_pred.size):
                ls.append('%s,%s,%s,%s,%s' % (x_test_id[i], pt_label[ls_test_y[i]], pt_label[_pred[i]], pt_label[_pred1[i]], np.count_nonzero(X_test[i,:,0] != -9999)))

            f_test = os.path.join(dd_validation, 'lr_test_acc_%s.csv' % _ty)
            with open(f_test, 'w') as _fo:
                _fo.write('\n'.join(ls))

    print (time.time()-a)

def main(opts):
    import os
    from gio import multi_task
    # _ts = [1,2,3,4,5,6,7,8,9,10]
    # multi_task.run(_task,  [_t for _t in multi_task.load(_ts, opts)], opts)
    # _task() for all samples and _task1() for sampled samples
    # _task()

    _task()
    
def usage():
    _p = environ_mag.usage(True)
    return _p

if __name__ == '__main__':
    from gio import environ_mag

    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
