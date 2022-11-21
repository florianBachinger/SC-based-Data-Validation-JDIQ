import numpy as np
import SyntheticError.Functions as fn
import random
from scipy.stats import multivariate_normal

class Univariate:
  @staticmethod
  def ApplyErrorFunction(error_function_name, target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    if(error_function_name == 'Square'):
      data_error,error = Univariate.Square(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif(error_function_name == 'Spike'):
      data_error,error = Univariate.Spike(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif(error_function_name == 'RampUp'):
      data_error,error = Univariate.RampUp(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif(error_function_name == 'RampDown'):
      data_error,error = Univariate.RampDown(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif(error_function_name == 'ExponentialUp'):
      data_error,error = Univariate.ExponentialUp(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif(error_function_name == 'Normal'):
      data_error,error = Univariate.Normal(target,start=start,end=end,error_value = error_value,returnErrorFunction=True)
    elif ((error_function_name == 'None') | (error_function_name == None)):
      data_error = target 
      error = [0]*len(target)
    else:
      raise f"error function with name {error_function_name} not supported"

    if(returnErrorFunction):
      return data_error, error
    else:
      return error

  @staticmethod
  def Square(target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    error = fn.Square(len(target),start,end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error

  @staticmethod
  def Spike(target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    error = fn.Spike(len(target),start,end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error

  @staticmethod
  def RampUp(target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    error = fn.RampUp(len(target),start,end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error

  @staticmethod
  def RampDown(target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    error = fn.RampDown(len(target),start,end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error

  @staticmethod
  def ExponentialUp(target,start=0,end = None, error_value = None, returnErrorFunction = False):
    if(error_value == None):
      error_value = np.std(target)
    error = fn.ExponentialUp(len(target),start,end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error

  @staticmethod
  def Normal(target,start=0,end = None, error_value = None, returnErrorFunction = False , stddev = 1,mean = 0,sample_start = -10, sample_end = 10):
    error = fn.Normal(len(target),start,end,stddev,mean,sample_start,sample_end)

    target_error = target + error_value * error
    if(returnErrorFunction):
      return target_error, error
    return target_error


class Multivariate:
  @staticmethod
  def CalculateAffectedSpace(data_length, dimension_input_space, error_width_percentage  ):
    number_inputs = len(dimension_input_space)

    # Calculate effective subset of given dimensions
    dimension_width =[(var['high'] - var['low']) for var in dimension_input_space] 
    full_volume = np.prod(dimension_width)
    volume_affected = full_volume * error_width_percentage

    # 10% affected data is not achieved by 10% of each dimension 
    # but 10% of dimension preserves ratio
    percent_of_dimension_width = [(var * error_width_percentage) for var in dimension_width] 
    aspect_volume = np.prod(percent_of_dimension_width)
    
    # see how many multiples of each dimension we need to get the calculated affected volume
    missing_volume_multiple = volume_affected/aspect_volume
    aspect_factor = missing_volume_multiple ** (1/number_inputs)

    dimension_part = [(var * aspect_factor) for var in percent_of_dimension_width]
    affected_volume_aspectRatio_preserved = np.prod(dimension_part)

    #assertions
    if ( np.abs(( affected_volume_aspectRatio_preserved - volume_affected) ) > 0.00001 ):
      raise f'affected volumes do not match'
    return dimension_part

  @staticmethod
  def ShiftAffectedSpace(dimension_data, affected_space):
    affected_space_positioned = []
    for (dimension_input,dimension_affected ) in zip(dimension_data,affected_space):
      if(dimension_input['width'] <dimension_affected):
        raise f"width of input dimension [{dimension_input['width']}] is smaller than affected space [{dimension_affected}]"
      
      halve_affected_width = dimension_affected/2
      center_of_affected = random.uniform(dimension_input['low'] + halve_affected_width, dimension_input['high'] - halve_affected_width)
      affected_space_positioned.append({
        'name' : dimension_input['name'],
        'low' : center_of_affected-halve_affected_width,
        'high' : center_of_affected+halve_affected_width,
        'width' : dimension_affected,
        'center' : center_of_affected
      })
    return affected_space_positioned

  @staticmethod
  def ApplyErrorFunction(error_function_name, data, input_target_name, output_target_name,error_function_variable_name, affected_space,  error_value = None,  returnErrorFunction = False):
    data[error_function_variable_name] = [0] * len(data)
    mask = np.ones(len(data))
    for shifted_space_dimension in affected_space:
      mask = mask & ((shifted_space_dimension['low']<= data[shifted_space_dimension['name']] )
                            & (data[shifted_space_dimension['name']] <= shifted_space_dimension['high']))

    if(np.sum(mask)==0):
      raise 'no data affected'

    centers = [var['center'] for var in affected_space]
    inputs = [var['name'] for var in affected_space]
    number_of_inputs = len(inputs)

    target = data[input_target_name]

    if(error_value == None):
      error_value = np.std(target)


    if(error_function_name == 'Square'):
      data.loc[mask,error_function_variable_name] = [1] * np.sum(mask)

    elif(error_function_name == 'Spike'):
      X = data.loc[mask,inputs].to_numpy()

      distance_center = [ np.sqrt(np.sum(np.square(item - centers ))) for item in X ]
      neg_distance = [ val*-1 for val in distance_center ]
      data.loc[mask,error_function_variable_name] = fn.Normalize01(neg_distance)
      
    elif(error_function_name == 'Normal'):
      X = data.loc[mask,inputs].to_numpy()
      rv = multivariate_normal(centers, np.diag(np.ones(number_of_inputs)))
      data.loc[mask,error_function_variable_name] = rv.pdf( X)

    elif ((error_function_name == 'None') | (error_function_name == None)):
      data[output_target_name] = target 
      data[error_function_variable_name]  = [0]*len(target)

    else:
      raise f"error function with name {error_function_name} not supported"

    data[output_target_name] = target +  data[error_function_variable_name] * error_value
    return data