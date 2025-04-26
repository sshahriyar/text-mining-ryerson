import darts.datasets
import numpy as np
import json


def get_dataset(dsname):
    darts_ds = getattr(darts.datasets,dsname)().load()
    if dsname=='GasRateCO2Dataset':
        darts_ds = darts_ds[darts_ds.columns[1]]
    series = darts_ds.pd_series()

    if dsname == 'SunspotsDataset':
        series = series.iloc[::4]
    if dsname =='HeartRateDataset':
        series = series.iloc[::2]
    return series

def get_datasets(n=-1,testfrac=0.2):
    datasets = [
        'AirPassengersDataset',
        'AusBeerDataset',
        'GasRateCO2Dataset', # multivariate
        'MonthlyMilkDataset',
        'SunspotsDataset', #very big, need to subsample?
        'WineDataset',
        'WoolyDataset',
        'HeartRateDataset',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        series = get_dataset(dsname)
        splitpoint = int(len(series)*(1-testfrac))
        
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))


if __name__ == '__main__':
    datasets = get_datasets()
    task_data = []
    for dsname,data in datasets.items():
        train, test = data
        task_data.append({
            'input': train.tolist(),
            'output': test.tolist()
        })
    with open('task.json', 'w', encoding='utf8') as f:
        json.dump(task_data, f, indent=4)
