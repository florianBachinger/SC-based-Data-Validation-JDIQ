import numpy as np
import scipy as sp

def Normalize01(x):
  return (x-min(x))/(max(x)-min(x))

def Square(length,start=0,end = None):
  if(end==None):
    end = length-1

  start = int(start)
  end = int(end)

  data = [0.0]*length
  errorLen = end-start
  data[start : end] = [1]* errorLen
  return np.array(data,dtype=np.float64)


def Spike(length,start=0,end = None):
  if(end==None):
    end = length-1
    
  start = int(start)
  end = int(end)

  data = [0.0]*length
  errorLen = end-start
  
  error = np.array([],dtype=np.float64)
  rampup = np.linspace(0,1,errorLen//2)
  rampdown = rampup[::-1] # reverse array
  if((len(rampup)+len(rampdown))<errorLen):
    error = np.append(rampup,[1])
    error = np.append(error,rampdown)
  else:
    error = np.append(rampup,rampdown)

  data[start : end] = error
  return np.array(data,dtype=np.float64)

def RampUp(length,start=0,end = None):
  if(end==None):
    end = length-1
    
  start = int(start)
  end = int(end)
  
  data = [0.0]*length
  data[start : length] = [1.0] * (length-start)
  
  errorLen = end-start
  rampup = np.linspace(0,1,errorLen)

  data[start : end] = rampup
  return np.array(data,dtype=np.float64)

def RampDown(length,start=0,end = None):
  if(end==None):
    end = length-1
    
  start = int(start)
  end = int(end)
  
  data = [0]*length
  data[start : length] = [-1] * (length-start)
  
  errorLen = end-start
  rampdown = np.linspace(0,-1,errorLen)

  data[start : end] = rampdown
  return np.array(data,dtype=np.float64)

def ExponentialUp(length,start=0,end = None, sample_start = -5, sample_end = 5 ):
  if(end==None):
    end = length-1
    
  start = int(start)
  end = int(end)
  
  data = [0.0]*length
  data[start : length] = [1.0] * (length-start)
  
  errorLen = end-start
  exp_sample = np.linspace(sample_start,sample_end,errorLen)

  data[start : end] = Normalize01(np.exp(exp_sample)) * 2
  return np.array(data,dtype=np.float64)

def NormalPDF(x, stddev = 1,mean = 0 ):
  return (1 / (stddev * np.sqrt(2*np.pi))) * np.exp((-1/2) * ((x-mean)/stddev)**2)
  
def Normal(length,start=0,end = None, stddev = 1,mean = 0,sample_start = -3, sample_end = 3 ):
  if(end==None):
    end = length-1
    
  start = int(start)
  end = int(end)
  
  data = [0.0]*length
  errorLen = end-start

  PDF_x_sample = np.linspace(sample_start,sample_end,errorLen)

  data[start : end] = Normalize01(NormalPDF(PDF_x_sample,stddev,mean)) 
  return np.array(data,dtype=np.float64)


# i.e. Area under the curve
# error = Square(2000,1600)
# print(np.sum(error))

# error = Spike(2000,1600)
# print(np.sum(error))

# error = RampUp(2000,1600)
# print(np.sum(error))

# error = RampDown(2000,1600)
# print(np.sum(error))

# error = ExponentialUp(2000,1600)
# print(np.sum(error))

# error = Normal(2000,1600)
# print(np.sum(error))