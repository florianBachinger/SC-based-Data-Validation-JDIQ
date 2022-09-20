import numpy as np
import SyntheticError.Functions as fn

def ApplyErrorFunction(error_function_name, target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  if(error_function_name == 'Square'):
    data_error,error = Square(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif(error_function_name == 'Spike'):
    data_error,error = Spike(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif(error_function_name == 'RampUp'):
    data_error,error = RampUp(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif(error_function_name == 'RampDown'):
    data_error,error = RampDown(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif(error_function_name == 'ExponentialUp'):
    data_error,error = ExponentialUp(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif(error_function_name == 'Normal'):
    data_error,error = Normal(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
  elif ((error_function_name == 'None') | (error_function_name == None)):
    data_error = target 
    error = [0]*len(target)
  else:
    raise f"error function with name {error_function_name} not supported"

  if(returnErrorFunction):
    return data_error, error
  else:
    return error

def Square(target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  error = fn.Square(len(target),start,end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def Spike(target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  error = fn.Spike(len(target),start,end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def RampUp(target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  error = fn.RampUp(len(target),start,end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def RampDown(target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  error = fn.RampDown(len(target),start,end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def ExponentialUp(target,start=0,end = None, error_value = None, returnErrorFunction = False):
  if(error_value == None):
    error_value = np.std(target)
  error = fn.ExponentialUp(len(target),start,end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error

def Normal(target,start=0,end = None, error_value = None, returnErrorFunction = False , stddev = 1,mean = 0,sample_start = -10, sample_end = 10):
  error = fn.Normal(len(target),start,end,stddev,mean,sample_start,sample_end)

  target_error = target + error_value * error
  if(returnErrorFunction):
    return target_error, error
  return target_error
