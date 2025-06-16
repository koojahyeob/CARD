import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def metric_by_var(pred: np.ndarray,
                  true: np.ndarray,
                  var_axis: int = 2):
    """
    채널/변수별 MSE·MAE 등을 반환
    pred, true : shape = (N, pred_len, C)

    Returns
    -------
    dict
        {
          'mae': (C,) ndarray,
          'mse': (C,) ndarray,
          'rmse':(C,) ndarray,
          'mape':(C,) ndarray,
          'mspe':(C,) ndarray
        }
    """
    # 배치축 + 시점축을 모두 평균하고, var_axis (채널) 만 남긴다
    reduce_axes = tuple(i for i in range(pred.ndim) if i != var_axis)

    mae  = np.mean(np.abs(pred - true),        axis=reduce_axes)
    mse  = np.mean(np.square(pred - true),     axis=reduce_axes)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true + 1e-8), axis=reduce_axes)
    mspe = np.mean(np.square((pred - true) / true + 1e-8), axis=reduce_axes)

    return dict(mae=mae, mse=mse, rmse=rmse, mape=mape, mspe=mspe)