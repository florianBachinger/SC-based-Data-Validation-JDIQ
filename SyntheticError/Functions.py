import numpy as np
import scipy as sp

def Normalize01(x):
  return (x-np.min(x))/(np.max(x)-np.min(x))

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

  if(errorLen == 1):
    PDF_x_sample = np.array([mean])
  if(errorLen == 2):
    PDF_x_sample = np.array([mean,mean])
  else:
    PDF_x_sample = np.linspace(sample_start,sample_end,errorLen)

  data[start : end] = NormalPDF(PDF_x_sample,stddev,mean)
  return Normalize01(np.array(data,dtype=np.float64))