import math
import numpy as np
import pandas as pd
import os
from lib_wetland import calculate_index
from gio import geo_raster_ex as gx
from gio import geo_base as gb
from gio import geo_raster as ge
import copy
import time
import numpy as np
from scipy.stats import chi2

bnd_idx = {'L30': {'B01':'Aerosol', 'B02':'Blue', 'B03':'Green', 'B04':'Red', 'B05':'NIR', 'B06':'SWIR1', 'B07':'SWIR2', 'Fmask':'Fmask'}, \
           'S30': {'B01':'Aerosol', 'B02':'Blue', 'B03':'Green', 'B04':'Red', 'B8A':'NIR', 'B11':'SWIR1', 'B12':'SWIR2', 'Fmask':'Fmask'}}

bnd_ids = ['Aerosol', 'Blue', 'Green','Red', 'NIR', 'SWIR1', 'SWIR2','NDVI', 'NDMI','NDWI','NDIIb7','EVI','SAVI','MSAVI']
_days = 16
_edge = math.ceil(60 / _days)

def calculate_indexes(pt):
    pt['NDVI'] = calculate_index.cal_index(pt['NIR'], pt['Red'])
    pt['NDMI'] = calculate_index.cal_index(pt['NIR'], pt['SWIR1'])
    pt['NDWI'] = calculate_index.cal_index(pt['Green'], pt['NIR'])
    pt['NDIIb7'] = calculate_index.cal_index(pt['NIR'], pt['SWIR2'])
    pt['EVI'] = calculate_index.cal_evi(pt['Blue'], pt['Red'], pt['NIR'])
    pt['SAVI'] = calculate_index.cal_savi(pt['Red'], pt['NIR'])
    pt['MSAVI'] = calculate_index.cal_msavi(pt['Red'], pt['NIR'])
    
def agg_interp_days(ps):
    _ms = {}
    for _d, _v in ps.items():
        _r = int(_d[:3]) // _days
        _ms[_r] = _ms.get(_r, [])
        _ms[_r].append(_v)

    _ds = {}
    for _r, _vs in _ms.items():
        if len(_vs) == 1:
            _ds[_r] = _vs[0]
            continue

        _vs = sorted(_vs, key=lambda x: x[4], reverse=True)
        _ds[_r] = _vs[int(len(_vs) / 2)]
       
    ps = {}
    for i in range(7):
        ls_vals = [np.nan for i in range(365 // _days)]
        for _r, _v in _ds.items():
            ls_vals[_r] = _v[i]

        ls_vals = ls_vals[_edge:-_edge]
        if ls_vals.count(np.nan) == len(ls_vals):
            return 0, 0
        else:
            _ts = pd.Series(ls_vals)
            _ts = _ts.interpolate(method="linear", limit_direction="both")
            ps[bnd_ids[i]] = list(_ts)

    return list(_ds.keys()), ps

def output_ts(pt, valid_dds, f_out):
    ls_agg = ['doy,timestep,' + ','.join(bnd_ids)]
    ls_interp = []
    for i in range(len(pt['NDVI'])):
        lt_a = []
        lt_i = []
        _doy = (_edge + i + 0.5) * _days
        _flag = _edge + i in valid_dds

        for _id in bnd_ids:
            lt_i.append(str(pt[_id][i]))
            if _flag:
                lt_a.append(str(pt[_id][i]))
            else:
                lt_a.append('-9999')
        ls_agg.append('%s,%s,%s' % (_doy, i, ','.join(lt_a)))
        ls_interp.append('%s,%s,%s' % (_doy, i, ','.join(lt_i)))

    ls_agg.extend(ls_interp)
    with open(f_out, 'w') as _fo:
        _fo.write('\n'.join(ls_agg))
        
def _task(_ff, d_out):
    ps = {}
    with open(_ff) as ff:
        for line in ff.read().splitlines()[1:]:
            _l = line.split(',')
            if _l[0][-3:] == 'S30':
                ps[_l[0]] = [float(_v) for _v in _l[1:]]

    f_out = os.path.join(d_out, '%s.csv' % os.path.basename(_ff)[:-8])
    
    valid_dds, pt_interp = agg_interp_days(ps)
    if pt_interp == 0:
        ls_agg = ['doy,timestep,' + ','.join(bnd_ids)]
        ls_vals = ['-9999' for i in range(365 // _days)][_edge:-_edge]
        for j in range(2):
            for i in range(len(ls_vals)):
                _doy = (_edge + i + 0.5) * _days
                ls_agg.append('%s,%s,%s' % (_doy,i,','.join(ls_vals)))
        with open(f_out, 'w') as _fo:
            _fo.write('\n'.join(ls_agg))
        return
        
    calculate_indexes(pt_interp)
    output_ts(pt_interp, valid_dds, f_out)
   
def main(opts):
    dd_sam_in = '/mnt/data_1/chexh/NSF_data/samples/ori'
    dd_sam_out = '/mnt/data_1/chexh/NSF_data/samples/LANDSAT/16d'

    _ts = []
    for _ff in os.listdir(dd_sam_in):
        _ts.append(os.path.join(dd_sam_in, _ff))

    from gio import multi_task
    multi_task.run(_task, [(_r, dd_sam_out) for _r in multi_task.load(_ts, opts)], opts)

def usage():
    _p = environ_mag.usage(True)
    return _p

if __name__ == '__main__':
    from gio import environ_mag

    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
