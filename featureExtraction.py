import pandas as pd
import numpy as np
from scipy import stats
from config import parse_args

args = parse_args()

filelist = ["ordinary.csv", "breath.csv", "swing.csv", "upp.csv", "out.csv"]


# coefficient_of_variation
def coefficient_of_variation(data):
    mean=np.mean(data) #计算平均值
    std=np.std(data,ddof=0) #计算标准差
    cv=std/mean
    return cv

for label_num, file in enumerate(filelist):
    print(label_num,file)
    data = pd.read_csv("./data/extract/" + file)
    data = data.to_numpy()

    label = [label_num for _ in range(len(data) - args.step_len)]


    ax_mean = []
    ay_mean = []
    az_mean = []
    wx_mean = []
    wy_mean = []
    wz_mean = []

    ax_var = []
    ay_var = []
    az_var = []
    wx_var = []
    wy_var = []
    wz_var = []

    ax_std = []
    ay_std = []
    az_std = []
    wx_std = []
    wy_std = []
    wz_std = []

    ax_rms = []
    ay_rms = []
    az_rms = []
    wx_rms = []
    wy_rms = []
    wz_rms = []

    ax_skew = []
    ay_skew = []
    az_skew = []
    wx_skew = []
    wy_skew = []
    wz_skew = []

    ax_kurt = []
    ay_kurt = []
    az_kurt = []
    wx_kurt = []
    wy_kurt = []
    wz_kurt = []

    ax_med = []
    ay_med = []
    az_med = []
    wx_med = []
    wy_med = []
    wz_med = []

    ax_min = []
    ay_min = []
    az_min = []
    wx_min = []
    wy_min = []
    wz_min = []

    ax_max = []
    ay_max = []
    az_max = []
    wx_max = []
    wy_max = []
    wz_max = []

    ax_per = []
    ay_per = []
    az_per = []
    wx_per = []
    wy_per = []
    wz_per = []
    
    ax_cv = []
    ay_cv = []
    az_cv = []
    wx_cv = []
    wy_cv = []
    wz_cv = []

    ax_ptp = []
    ay_ptp = []
    az_ptp = []
    wx_ptp = []
    wy_ptp = []
    wz_ptp = []


    for i in range(0, len(data) - args.step_len):
        ax_data = data[i:i+args.step_len-1][0]
        ay_data = data[i:i+args.step_len-1][1]
        az_data = data[i:i+args.step_len-1][2]
        wx_data = data[i:i+args.step_len-1][3]
        wy_data = data[i:i+args.step_len-1][4]
        wz_data = data[i:i+args.step_len-1][5]

        ax_mean.append(ax_data.mean())
        ay_mean.append(ay_data.mean())
        az_mean.append(az_data.mean())
        wx_mean.append(wx_data.mean())
        wy_mean.append(wy_data.mean())
        wz_mean.append(wz_data.mean())

        ax_var.append(ax_data.var())
        ay_var.append(ay_data.var())
        az_var.append(az_data.var())
        wx_var.append(wx_data.var())
        wy_var.append(wy_data.var())
        wz_var.append(wz_data.var())

        ax_std.append(ax_data.std())
        ay_std.append(ay_data.std())
        az_std.append(az_data.std())
        wx_std.append(wx_data.std())
        wy_std.append(wy_data.std())
        wz_std.append(wz_data.std())

        ax_rms.append(np.sqrt(np.mean(np.square(ax_data))))
        ay_rms.append(np.sqrt(np.mean(np.square(ay_data))))
        az_rms.append(np.sqrt(np.mean(np.square(az_data))))
        wx_rms.append(np.sqrt(np.mean(np.square(wx_data))))
        wy_rms.append(np.sqrt(np.mean(np.square(wy_data))))
        wz_rms.append(np.sqrt(np.mean(np.square(wz_data))))

        ax_skew.append(stats.skew(ax_data))
        ay_skew.append(stats.skew(ay_data))
        az_skew.append(stats.skew(az_data))
        wx_skew.append(stats.skew(wx_data))
        wy_skew.append(stats.skew(wy_data))
        wz_skew.append(stats.skew(wz_data))

        ax_kurt.append(stats.kurtosis(ax_data))
        ay_kurt.append(stats.kurtosis(ay_data))
        az_kurt.append(stats.kurtosis(az_data))
        wx_kurt.append(stats.kurtosis(wx_data))
        wy_kurt.append(stats.kurtosis(wy_data))
        wz_kurt.append(stats.kurtosis(wz_data))
        
        ax_med.append(np.median(ax_data))
        ay_med.append(np.median(ay_data))
        az_med.append(np.median(az_data))
        wx_med.append(np.median(wx_data))
        wy_med.append(np.median(wy_data))
        wz_med.append(np.median(wz_data))

        ax_min.append(np.min(ax_data))
        ay_min.append(np.min(ay_data))
        az_min.append(np.min(az_data))
        wx_min.append(np.min(wx_data))
        wy_min.append(np.min(wy_data))
        wz_min.append(np.min(wz_data))

        ax_max.append(np.max(ax_data))
        ay_max.append(np.max(ay_data))
        az_max.append(np.max(az_data))
        wx_max.append(np.max(wx_data))
        wy_max.append(np.max(wy_data))
        wz_max.append(np.max(wz_data))

        ax_per.append(np.percentile(ax_data, 75) - np.percentile(ax_data, 25))
        ay_per.append(np.percentile(ay_data, 75) - np.percentile(ay_data, 25))
        az_per.append(np.percentile(az_data, 75) - np.percentile(az_data, 25))
        wx_per.append(np.percentile(wx_data, 75) - np.percentile(wx_data, 25))
        wy_per.append(np.percentile(wy_data, 75) - np.percentile(wy_data, 25))
        wz_per.append(np.percentile(wz_data, 75) - np.percentile(wz_data, 25))

        ax_cv.append(coefficient_of_variation(ax_data))
        ay_cv.append(coefficient_of_variation(ay_data))
        az_cv.append(coefficient_of_variation(az_data))
        wx_cv.append(coefficient_of_variation(wx_data))
        wy_cv.append(coefficient_of_variation(wy_data))
        wz_cv.append(coefficient_of_variation(wz_data))

        ax_ptp.append(np.ptp(ax_data))
        ay_ptp.append(np.ptp(ay_data))
        az_ptp.append(np.ptp(az_data))
        wx_ptp.append(np.ptp(wx_data))
        wy_ptp.append(np.ptp(wy_data))
        wz_ptp.append(np.ptp(wz_data))

    # print(mean)

    df = pd.DataFrame({'ax_mean':ax_mean,
                        'ay_mean':ay_mean,
                        'az_mean':az_mean,
                        'wx_mean':wx_mean,
                        'wy_mean':wy_mean,
                        'wz_mean':wz_mean,
                        'ax_var':ax_var,
                        'ay_var':ay_var,
                        'az_var':az_var,
                        'wx_var':wx_var,
                        'wy_var':wy_var,
                        'wz_var':wz_var,
                        'ax_std':ax_std,
                        'ay_std':ay_std,
                        'az_std':az_std,
                        'wx_std':wx_std,
                        'wy_std':wy_std,
                        'wz_std':wz_std,
                        'ax_rms':ax_rms,
                        'ay_rms':ay_rms,
                        'az_rms':az_rms,
                        'wx_rms':wx_rms,
                        'wy_rms':wy_rms,
                        'wz_rms':wz_rms,
                        'ax_skew':ax_skew,
                        'ay_skew':ay_skew,
                        'az_skew':az_skew,
                        'wx_skew':wx_skew,
                        'wy_skew':wy_skew,
                        'wz_skew':wz_skew,
                        'ax_kurt':ax_kurt,
                        'ay_kurt':ay_kurt,
                        'az_kurt':az_kurt,
                        'wx_kurt':wx_kurt,
                        'wy_kurt':wy_kurt,
                        'wz_kurt':wz_kurt,
                        'ax_med':ax_med,
                        'ay_med':ay_med,
                        'az_med':az_med,
                        'wx_med':wx_med,
                        'wy_med':wy_med,
                        'wz_med':wz_med,
                        'ax_min':ax_min,
                        'ay_min':ay_min,
                        'az_min':az_min,
                        'wx_min':wx_min,
                        'wy_min':wy_min,
                        'wz_min':wz_min,
                        'ax_max':ax_max,
                        'ay_max':ay_max,
                        'az_max':az_max,
                        'wx_max':wx_max,
                        'wy_max':wy_max,
                        'wz_max':wz_max,
                        'ax_per':ax_per,
                        'ay_per':ay_per,
                        'az_per':az_per,
                        'wx_per':wx_per,
                        'wy_per':wy_per,
                        'wz_per':wz_per,
                        'ax_cv':ax_cv,
                        'ay_cv':ay_cv,
                        'az_cv':az_cv,
                        'wx_cv':wx_cv,
                        'wy_cv':wy_cv,
                        'wz_cv':wz_cv,
                        'ax_ptp':ax_ptp,
                        'ay_ptp':ay_ptp,
                        'az_ptp':az_ptp,
                        'wx_ptp':wx_ptp,
                        'wy_ptp':wy_ptp,
                        'wz_ptp':wz_ptp,
                        'label':label}).to_csv("./data/feature/" + file)