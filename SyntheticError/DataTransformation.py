import numpy as np
import SyntheticError.Functions as fn


def Square(target,start=0,end = None, returnErrorFunction = False):
  error = fn.Square(len(target),start,end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def Spike(target,start=0,end = None, returnErrorFunction = False):
  error = fn.Spike(len(target),start,end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def RampUp(target,start=0,end = None, returnErrorFunction = False):
  error = fn.RampUp(len(target),start,end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def RampDown(target,start=0,end = None, returnErrorFunction = False):
  error = fn.RampDown(len(target),start,end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def ExponentialUp(target,start=0,end = None, returnErrorFunction = False):
  error = fn.ExponentialUp(len(target),start,end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def Normal(target,start=0,end = None, returnErrorFunction = False , stddev = 1,mean = 0,sample_start = -10, sample_end = 10):
  error = fn.Normal(len(target),start,end,stddev,mean,sample_start,sample_end)

  target_error = target + np.std(target) * error
  if(returnErrorFunction):
    return target_error, error
  return target_error
