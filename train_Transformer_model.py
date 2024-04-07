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
from lib_wetland import transformer
import time
import pickle
import logging
from sklearn.model_selection import train_test_split
from collections import defaultdict

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def parse_file(_ff):
    ls_agg_x = []
    ls_interp_x = []
    with open(_ff, 'r') as _f_in:
        _info = _f_in.read().splitlines()
        for line in _info[1:15]:
            _l = line.split(',')
            ls_agg_x.append([float(_v) for _v in _l[2:]])

        for line in _info[15:]:
            _l = line.split(',')
            ls_interp_x.append([float(_v) for _v in _l[2:]])

    return np.array(ls_agg_x), np.array(ls_interp_x)


def stat(y):
    ps = {}
    for _v in y:
        ps[_v+1] = ps.get(_v+1, 0)
        ps[_v + 1] = ps[_v + 1] +1
    return len(y), ps.items()

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

def filter_test_id(x, x_id):
    x_test_id = []
    _num = x.shape[0]
    for i in range(_num):
        if np.sum(x[i, :, 0] == -9999) == 14:
            continue
        x_test_id.append(x_id[i])
    return x_test_id
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


def _task1(_version):

    dd_sam = '/mnt/data_1/chexh/NSF_data/samples/random_samples/16d_rand_v%s' % _version
    dd_model = '/mnt/data_1/chexh/NSF_data/result_transform_random/models/paras/v%s' % _version
    dd_hist = '/mnt/data_1/chexh/NSF_data/result_transform_random/models/hist/v%s' % _version
    dd_norm = '/mnt/data_1/chexh/NSF_data/result_transform_random/norm/v%s' % _version
    dd_validation = '/mnt/data_1/chexh/NSF_data/result_transform_random/validation/v%s' % _version

    os.path.exists(dd_model) or os.makedirs(dd_model)
    os.path.exists(dd_hist) or os.makedirs(dd_hist)
    os.path.exists(dd_norm) or os.makedirs(dd_norm)
    os.path.exists(dd_validation) or os.makedirs(dd_validation)

    ls_agg_x = []
    ls_interp_x = []
    ls_y = []
    _ffs = os.listdir(dd_sam)
    _ffs.sort()
    for _ff in _ffs:
        _m = re.match('id-(.*)_lc-(\d)\.csv$', _ff)
        if _m:
            _id = int(_m.group(1))
            _ty = int(_m.group(2))
            _agg_x, _interp_x = parse_file(os.path.join(dd_sam, _ff))
            ls_agg_x.append([_agg_x, _id])
            ls_interp_x.append([_interp_x, _id])
            ls_y.append([_ty - 1, _id])

    x_mean_agg, x_std_agg = output_norm(ls_agg_x, dd_norm, 'mask')
    x_mean_interp, x_std_interp = output_norm(ls_interp_x, dd_norm, 'interp')

    a = time.time()
    # for _version in [1,2,3,4,5]:
    for _ver in [1]:
        for _seed in range(5):

            _ty = '16d_%s' % _seed
            sample_tys = '/mnt/data_1/chexh/NSF_data/samples/HLS_sample_ids/fold%s_samples_ids.csv' % _seed
            pt_tys = defaultdict(list)
            for _str in open(sample_tys).read().splitlines()[1:]:
                _str = _str.split(',')
                pt_tys[_str[1]].append(int(_str[0]))

            X_train = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['train']]
            X_train = normlize(X_train, x_mean_agg, x_std_agg)
            ls_train_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test = normlize(X_test, x_mean_agg, x_std_agg)
            ls_test_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]


            X_test_ids = [_vv[1] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test_ids = filter_test_id(X_test, X_test_ids)

            X_validation = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['validation']]
            X_validation = normlize(X_validation, x_mean_agg, x_std_agg)
            ls_validation_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]


            X_train1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['train'] ]
            X_train1 = normlize(X_train1, x_mean_interp, x_std_interp)
            ls_train_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['test']]
            X_test1 = normlize(X_test1, x_mean_interp, x_std_interp)
            ls_test_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]

            X_validation1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['validation']]
            X_validation1= normlize(X_validation1, x_mean_interp, x_std_interp)
            ls_validation_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]


            X_train, ls_train_y = filter_invalid(X_train, ls_train_y)
            X_test, ls_test_y = filter_invalid(X_test, ls_test_y)
            X_validation, ls_validation_y = filter_invalid(X_validation, ls_validation_y)

            X_train1, ls_train_y1 = filter_invalid(X_train1, ls_train_y1)
            X_test1, ls_test_y1 = filter_invalid(X_test1, ls_test_y1)
            X_validation1, ls_validation_y1 = filter_invalid(X_validation1, ls_validation_y1)

            dd_model_msk = os.path.join(dd_model, 'lr_mask_%s.h5' % _ty)
            dd_model_int = os.path.join(dd_model, 'lr_interp_%s.h5' % _ty)

            train_n = X_train.shape[0]
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, np.array(ls_train_y))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, np.array(ls_validation_y))).batch(
                X_validation.shape[0])

            train_n = X_train1.shape[0]
            train_ds1 = tf.data.Dataset.from_tensor_slices((X_train1, np.array(ls_train_y1))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds1 = tf.data.Dataset.from_tensor_slices((X_validation1, np.array(ls_validation_y1))).batch(
                X_validation1.shape[0])


            transformer.trainings_val_class(train_ds, validation_ds, dd_model_msk,
                                            os.path.join(dd_hist, 'lr_mask_%s.csv' % _ty), EPOCH, 0.0001,
                                            X_train.shape[0])
            transformer.trainings_val_class(train_ds1,validation_ds1, dd_model_int, os.path.join(dd_hist, 'lr_interp_%s.csv' % _ty), EPOCH, 0.0001, X_train.shape[0])

            _model_msk = tf.keras.models.load_model(dd_model_msk)
            X_test = X_test.astype(np.float32)
            _pred = _model_msk(X_test, training=False)
            _pred = tf.argmax(_pred, 1)
            _pred = np.array(_pred).reshape((X_test.shape[0],))

            _model_int = tf.keras.models.load_model(dd_model_int)
            X_test1 = X_test1.astype(np.float32)
            _pred1 = _model_int(X_test1, training=False)
            _pred1 = tf.argmax(_pred1, 1)
            _pred1 = np.array(_pred1).reshape((X_test1.shape[0],))

            ls = ['id,agg,agg_pred,int_pred,valid_obs']
            for i in range(_pred.size):
                ls.append('%s,%s,%s,%s,%s' % (X_test_ids[i], ls_test_y[i], _pred[i], _pred1[i], np.count_nonzero(X_test[i,:,0] != -9999)))

            f_test = os.path.join(dd_validation, 'lr_test_acc_%s.csv' % _ty)
            with open(f_test, 'w') as _fo:
                _fo.write('\n'.join(ls))
    print (time.time() - a)

def _task(_sensor):

    using_kt = False
    dd_sam = '/mnt/data_1/chexh/NSF_data/samples/%s/16d' % _sensor
    dd_model = '/mnt/data_1/chexh/NSF_data/result_transform/models/paras/%s' % _sensor
    dd_hist = '/mnt/data_1/chexh/NSF_data/result_transform/models/hist/%s' % _sensor
    dd_norm = '/mnt/data_1/chexh/NSF_data/result_transform/norm/%s' % _sensor
    dd_validation = '/mnt/data_1/chexh/NSF_data/result_transform/validation/%s' % _sensor

    # dd_model = '/mnt/data_1/chexh/NSF_data/test1'
    # dd_hist = '/mnt/data_1/chexh/NSF_data/test1'
    # dd_norm = '/mnt/data_1/chexh/NSF_data/test1'
    # dd_validation = '/mnt/data_1/chexh/NSF_data/test1'


    os.path.exists(dd_model) or os.makedirs(dd_model)
    os.path.exists(dd_hist) or os.makedirs(dd_hist)
    os.path.exists(dd_norm) or os.makedirs(dd_norm)
    os.path.exists(dd_validation) or os.makedirs(dd_validation)
    
    ls_agg_x = []
    ls_interp_x = []
    ls_y = []
    _ffs = os.listdir(dd_sam)
    _ffs.sort()
    for _ff in _ffs:
        _m = re.match('id-(.*)_lc-(\d)\.csv$', _ff)
        if _m:
            _id = _m.group(1)
            _ty = int(_m.group(2))
            _agg_x, _interp_x = parse_file(os.path.join(dd_sam, _ff))
            ls_agg_x.append((_agg_x, _id))
            ls_interp_x.append((_interp_x, _id))
            ls_y.append(_ty-1)

    a = time.time()
    for _version in [1,2,3,4,5]:
        for _seed in range(5):

            _ty = '16d_%s_v%s' % (_seed, _version)

            X_train, X_test, X_validation, ls_train_y, ls_test_y, ls_validation_y, x_mean, x_std, x_test_id = sampling(ls_agg_x,  ls_y, _seed, dd_norm, '16d_mask')
            X_train1, X_test1, X_validation1, ls_train_y1, ls_test_y1, ls_validation_y1, x_mean1, x_std1, x_test_id1 = sampling(ls_interp_x, ls_y, _seed, dd_norm, '16d_interp')

            X_train, ls_train_y = filter_invalid(X_train, ls_train_y)
            X_test, ls_test_y = filter_invalid(X_test, ls_test_y)
            X_validation, ls_validation_y = filter_invalid(X_validation, ls_validation_y)

            X_train1, ls_train_y1 = filter_invalid(X_train1, ls_train_y1)
            X_test1, ls_test_y1 = filter_invalid(X_test1, ls_test_y1)
            X_validation1, ls_validation_y1 = filter_invalid(X_validation1, ls_validation_y1)

            # X_train = X_train[:,:,1:7]
            # X_test = X_test[:,:,1:7]
            # X_validation = X_validation[:,:,1:7]
            # X_train1 = X_train1[:, :, 1:7]
            # X_test1 = X_test1[:, :, 1:7]
            # X_validation1 = X_validation1[:, :, 1:7]

            
            if using_kt:
                dd_model_msk = os.path.join(dd_model, 'kt_mask_%s' % _ty)
                os.path.exists(dd_model_msk) or os.makedirs(dd_model_msk)
                dd_model_int= os.path.join(dd_model, 'kt_interp_%s' % _ty)
                os.path.exists(dd_model_int) or os.makedirs(dd_model_int)
                LSTM_model.search_best_model_hy(_ty, X_train, np.array(ls_train_y),X_validation, np.array(ls_validation_y), \
                                             dd_model_msk, os.path.join(dd_hist, 'kt_mask_%s.csv' % _ty), True)
                LSTM_model.search_best_model_hy(_ty, X_train1, np.array(ls_train_y1), X_validation1, np.array(ls_validation_y1),\
                                             dd_model_int, os.path.join(dd_hist, 'kt_interp_%s.csv' % _ty), False)
            else:

                dd_model_msk = os.path.join(dd_model, 'lr_mask_%s.h5' % _ty)
                dd_model_int = os.path.join(dd_model, 'lr_interp_%s.h5' % _ty)

                train_n = X_train.shape[0]
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, np.array(ls_train_y))).shuffle(train_n + 1).batch(batch_size)
                validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, np.array(ls_validation_y))).batch(X_validation.shape[0])


                train_n = X_train1.shape[0]
                train_ds1 = tf.data.Dataset.from_tensor_slices((X_train1, np.array(ls_train_y1))).shuffle(train_n + 1).batch(batch_size)
                validation_ds1 = tf.data.Dataset.from_tensor_slices((X_validation1, np.array(ls_validation_y1))).batch(X_validation1.shape[0])

                # transformer.test(X_train, 14, 14, 8, 3, 64, 4, 0.1, True, -9999.0, "softmax", False, 0)
                # return

                transformer.trainings_val_class(train_ds, validation_ds, dd_model_msk, os.path.join(dd_hist, 'lr_mask_%s.csv' % _ty), EPOCH,  0.0001, X_train.shape[0])
                # transformer.trainings_val_class(train_ds1,validation_ds1, dd_model_int, os.path.join(dd_hist, 'lr_interp_%s.csv' % _ty), EPOCH, 0.0001, X_train.shape[0])

                # _model_msk = tf.keras.models.load_model(dd_model_msk)
                # X_test = X_test.astype(np.float32)
                # _pred = _model_msk(X_test, training=False)
                # _pred = tf.argmax(_pred, 1)
                # _pred = np.array(_pred).reshape((X_test.shape[0],))
                #
                # _model_int = tf.keras.models.load_model(dd_model_int)
                # X_test1 = X_test1.astype(np.float32)
                # _pred1 = _model_int(X_test1, training=False)
                # _pred1 = tf.argmax(_pred1, 1)
                # _pred1 = np.array(_pred1).reshape((X_test1.shape[0],))
                #
                # ls = ['id,agg,agg_pred,int_pred,valid_obs']
                # for i in range(_pred.size):
                #     ls.append('%s,%s,%s,%s,%s' % (x_test_id[i], ls_test_y[i], _pred[i], _pred1[i], np.count_nonzero(X_test[i,:,0] != -9999)))
                #
                # f_test = os.path.join(dd_validation, 'lr_test_acc_%s.csv' % _ty)
                # with open(f_test, 'w') as _fo:
                #     _fo.write('\n'.join(ls))
    print (time.time()-a)


def _task2(_sensor):
    using_kt = False
    dd_sam = '/mnt/data_1/chexh/NSF_data/samples/%s/16d' % _sensor
    dd_model = '/mnt/data_1/chexh/NSF_data/result_transform_filter/models/paras/%s' % _sensor
    dd_hist = '/mnt/data_1/chexh/NSF_data/result_transform_filter/models/hist/%s' % _sensor
    dd_norm = '/mnt/data_1/chexh/NSF_data/result_transform_filter/norm/%s' % _sensor
    dd_validation = '/mnt/data_1/chexh/NSF_data/result_transform_filter/validation/%s' % _sensor

    os.path.exists(dd_model) or os.makedirs(dd_model)
    os.path.exists(dd_hist) or os.makedirs(dd_hist)
    os.path.exists(dd_norm) or os.makedirs(dd_norm)
    os.path.exists(dd_validation) or os.makedirs(dd_validation)

    # 1—沼泽；2—水田 ；3—人工地表； 4—旱地；5—盐碱地；6—林地；7—明水面；8—草地

    ps_ty = {0:0, 1:1, 3:2, 5:3, 7:4}
    ps_ty_opp = {0:0, 1:1, 2:3, 3:5, 4:7}

    ls_agg_x = []
    ls_interp_x = []
    ls_y = []
    _ffs = os.listdir(dd_sam)
    _ffs.sort()
    for _ff in _ffs:
        _m = re.match('id-(.*)_lc-(\d)\.csv$', _ff)
        if _m:
            _id = int(_m.group(1))
            _ty = int(_m.group(2))
            if _ty in [3,5,7]:
                continue
            _agg_x, _interp_x = parse_file(os.path.join(dd_sam, _ff))
            ls_agg_x.append([_agg_x, _id])
            ls_interp_x.append([_interp_x, _id])
            ls_y.append([ps_ty[_ty - 1], _id])

    x_mean_agg, x_std_agg = output_norm(ls_agg_x, dd_norm, 'mask')
    x_mean_interp, x_std_interp = output_norm(ls_interp_x, dd_norm, 'interp')

    a = time.time()
    for _version in [1, 2, 3, 4, 5]:
        for _seed in range(5):

            _ty = '16d_%s_v%s' % (_seed, _version)

            sample_tys = '/mnt/data_1/chexh/NSF_data/samples/HLS_sample_ids/fold%s_samples_ids.csv' % _seed
            pt_tys = defaultdict(list)
            for _str in open(sample_tys).read().splitlines()[1:]:
                _str = _str.split(',')
                pt_tys[_str[1]].append(int(_str[0]))

            X_train = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['train']]
            X_train = normlize(X_train, x_mean_agg, x_std_agg)
            ls_train_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test = normlize(X_test, x_mean_agg, x_std_agg)
            ls_test_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]

            X_test_ids = [_vv[1] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test_ids = filter_test_id(X_test, X_test_ids)

            X_validation = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['validation']]
            X_validation = normlize(X_validation, x_mean_agg, x_std_agg)
            ls_validation_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]

            X_train1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['train']]
            X_train1 = normlize(X_train1, x_mean_interp, x_std_interp)
            ls_train_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['test']]
            X_test1 = normlize(X_test1, x_mean_interp, x_std_interp)
            ls_test_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]

            X_validation1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['validation']]
            X_validation1 = normlize(X_validation1, x_mean_interp, x_std_interp)
            ls_validation_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]

            X_train, ls_train_y = filter_invalid(X_train, ls_train_y)
            X_test, ls_test_y = filter_invalid(X_test, ls_test_y)
            X_validation, ls_validation_y = filter_invalid(X_validation, ls_validation_y)

            X_train1, ls_train_y1 = filter_invalid(X_train1, ls_train_y1)
            X_test1, ls_test_y1 = filter_invalid(X_test1, ls_test_y1)
            X_validation1, ls_validation_y1 = filter_invalid(X_validation1, ls_validation_y1)

            dd_model_msk = os.path.join(dd_model, 'lr_mask_%s.h5' % _ty)
            dd_model_int = os.path.join(dd_model, 'lr_interp_%s.h5' % _ty)

            train_n = X_train.shape[0]
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, np.array(ls_train_y))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, np.array(ls_validation_y))).batch(
                X_validation.shape[0])

            train_n = X_train1.shape[0]
            train_ds1 = tf.data.Dataset.from_tensor_slices((X_train1, np.array(ls_train_y1))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds1 = tf.data.Dataset.from_tensor_slices((X_validation1, np.array(ls_validation_y1))).batch(
                X_validation1.shape[0])

            transformer.trainings_val_class(train_ds, validation_ds, dd_model_msk,
                                            os.path.join(dd_hist, 'lr_mask_%s.csv' % _ty), EPOCH, 0.0001,
                                            X_train.shape[0])
            transformer.trainings_val_class(train_ds1, validation_ds1, dd_model_int,
                                            os.path.join(dd_hist, 'lr_interp_%s.csv' % _ty), EPOCH, 0.0001,
                                            X_train.shape[0])

            _model_msk = tf.keras.models.load_model(dd_model_msk)
            X_test = X_test.astype(np.float32)
            _pred = _model_msk(X_test, training=False)
            _pred = tf.argmax(_pred, 1)
            _pred = np.array(_pred).reshape((X_test.shape[0],))

            _model_int = tf.keras.models.load_model(dd_model_int)
            X_test1 = X_test1.astype(np.float32)
            _pred1 = _model_int(X_test1, training=False)
            _pred1 = tf.argmax(_pred1, 1)
            _pred1 = np.array(_pred1).reshape((X_test1.shape[0],))

            ls = ['id,agg,agg_pred,int_pred,valid_obs']
            for i in range(_pred.size):
                ls.append('%s,%s,%s,%s,%s' % (X_test_ids[i], ps_ty_opp[ls_test_y[i]], ps_ty_opp[_pred[i]], ps_ty_opp[_pred1[i]], np.count_nonzero(X_test[i, :, 0] != -9999)))

            f_test = os.path.join(dd_validation, 'lr_test_acc_%s.csv' % _ty)
            with open(f_test, 'w') as _fo:
                _fo.write('\n'.join(ls))

    print (time.time() - a)


def _task3(_version):

    dd_sam = '/mnt/data_1/chexh/NSF_data/samples/random_samples/16d_rand_v%s' % _version
    dd_model = '/mnt/data_1/chexh/NSF_data/result_transform_filter_random/models/paras/v%s' % _version
    dd_hist = '/mnt/data_1/chexh/NSF_data/result_transform_filter_random/models/hist/v%s' % _version
    dd_norm = '/mnt/data_1/chexh/NSF_data/result_transform_filter_random/norm/v%s' % _version
    dd_validation = '/mnt/data_1/chexh/NSF_data/result_transform_filter_random/validation/v%s' % _version

    os.path.exists(dd_model) or os.makedirs(dd_model)
    os.path.exists(dd_hist) or os.makedirs(dd_hist)
    os.path.exists(dd_norm) or os.makedirs(dd_norm)
    os.path.exists(dd_validation) or os.makedirs(dd_validation)

    # 1—沼泽；2—水田 ；3—人工地表； 4—旱地；5—盐碱地；6—林地；7—明水面；8—草地

    ps_ty = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4}
    ps_ty_opp = {0: 0, 1: 1, 2: 3, 3: 5, 4: 7}

    ls_agg_x = []
    ls_interp_x = []
    ls_y = []
    _ffs = os.listdir(dd_sam)
    _ffs.sort()
    for _ff in _ffs:
        _m = re.match('id-(.*)_lc-(\d)\.csv$', _ff)
        if _m:
            _id = int(_m.group(1))
            _ty = int(_m.group(2))
            if _ty in [3, 5, 7]:
                continue
            _agg_x, _interp_x = parse_file(os.path.join(dd_sam, _ff))
            ls_agg_x.append([_agg_x, _id])
            ls_interp_x.append([_interp_x, _id])
            ls_y.append([ps_ty[_ty - 1], _id])

    x_mean_agg, x_std_agg = output_norm(ls_agg_x, dd_norm, 'mask')
    x_mean_interp, x_std_interp = output_norm(ls_interp_x, dd_norm, 'interp')

    a = time.time()
    # for _version in [1,2,3,4,5]:
    for _ver in [1]:
        for _seed in range(5):

            _ty = '16d_%s' % _seed
            sample_tys = '/mnt/data_1/chexh/NSF_data/samples/HLS_sample_ids/fold%s_samples_ids.csv' % _seed
            pt_tys = defaultdict(list)
            for _str in open(sample_tys).read().splitlines()[1:]:
                _str = _str.split(',')
                pt_tys[_str[1]].append(int(_str[0]))

            X_train = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['train']]
            X_train = normlize(X_train, x_mean_agg, x_std_agg)
            ls_train_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test = normlize(X_test, x_mean_agg, x_std_agg)
            ls_test_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]

            X_test_ids = [_vv[1] for _vv in ls_agg_x if _vv[1] in pt_tys['test']]
            X_test_ids = filter_test_id(X_test, X_test_ids)

            X_validation = [_vv[0] for _vv in ls_agg_x if _vv[1] in pt_tys['validation']]
            X_validation = normlize(X_validation, x_mean_agg, x_std_agg)
            ls_validation_y = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]

            X_train1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['train']]
            X_train1 = normlize(X_train1, x_mean_interp, x_std_interp)
            ls_train_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['train']]

            X_test1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['test']]
            X_test1 = normlize(X_test1, x_mean_interp, x_std_interp)
            ls_test_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['test']]

            X_validation1 = [_vv[0] for _vv in ls_interp_x if _vv[1] in pt_tys['validation']]
            X_validation1 = normlize(X_validation1, x_mean_interp, x_std_interp)
            ls_validation_y1 = [_vv[0] for _vv in ls_y if _vv[1] in pt_tys['validation']]

            X_train, ls_train_y = filter_invalid(X_train, ls_train_y)
            X_test, ls_test_y = filter_invalid(X_test, ls_test_y)
            X_validation, ls_validation_y = filter_invalid(X_validation, ls_validation_y)

            X_train1, ls_train_y1 = filter_invalid(X_train1, ls_train_y1)
            X_test1, ls_test_y1 = filter_invalid(X_test1, ls_test_y1)
            X_validation1, ls_validation_y1 = filter_invalid(X_validation1, ls_validation_y1)

            dd_model_msk = os.path.join(dd_model, 'lr_mask_%s.h5' % _ty)
            dd_model_int = os.path.join(dd_model, 'lr_interp_%s.h5' % _ty)

            train_n = X_train.shape[0]
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, np.array(ls_train_y))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, np.array(ls_validation_y))).batch(
                X_validation.shape[0])

            train_n = X_train1.shape[0]
            train_ds1 = tf.data.Dataset.from_tensor_slices((X_train1, np.array(ls_train_y1))).shuffle(
                train_n + 1).batch(batch_size)
            validation_ds1 = tf.data.Dataset.from_tensor_slices((X_validation1, np.array(ls_validation_y1))).batch(
                X_validation1.shape[0])

            transformer.trainings_val_class(train_ds, validation_ds, dd_model_msk,
                                            os.path.join(dd_hist, 'lr_mask_%s.csv' % _ty), EPOCH, 0.0001,
                                            X_train.shape[0])
            transformer.trainings_val_class(train_ds1, validation_ds1, dd_model_int,
                                            os.path.join(dd_hist, 'lr_interp_%s.csv' % _ty), EPOCH, 0.0001,
                                            X_train.shape[0])

            _model_msk = tf.keras.models.load_model(dd_model_msk)
            X_test = X_test.astype(np.float32)
            _pred = _model_msk(X_test, training=False)
            _pred = tf.argmax(_pred, 1)
            _pred = np.array(_pred).reshape((X_test.shape[0],))

            _model_int = tf.keras.models.load_model(dd_model_int)
            X_test1 = X_test1.astype(np.float32)
            _pred1 = _model_int(X_test1, training=False)
            _pred1 = tf.argmax(_pred1, 1)
            _pred1 = np.array(_pred1).reshape((X_test1.shape[0],))

            ls = ['id,agg,agg_pred,int_pred,valid_obs']
            for i in range(_pred.size):
                ls.append('%s,%s,%s,%s,%s' % (X_test_ids[i], ps_ty_opp[ls_test_y[i]], ps_ty_opp[_pred[i]], ps_ty_opp[_pred1[i]], np.count_nonzero(X_test[i, :, 0] != -9999)))

            f_test = os.path.join(dd_validation, 'lr_test_acc_%s.csv' % _ty)
            with open(f_test, 'w') as _fo:
                _fo.write('\n'.join(ls))
    print (time.time() - a)


def main(opts):

    # _task() for all samples
    # _task1() for sampled samples
    # _task2() for all samples except for barren land, built-up, water body
    # _task3() for sampled samples except for barren land, built-up, water body

    import os
    from gio import multi_task

    _ts = [0,1,2,3,4]
    # multi_task.run(_task1, [_t for _t in multi_task.load(_ts, opts)], opts)
    multi_task.run(_task3, [_t for _t in multi_task.load(_ts, opts)], opts)

def usage():
    _p = environ_mag.usage(True)
    return _p

if __name__ == '__main__':
    from gio import environ_mag

    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
