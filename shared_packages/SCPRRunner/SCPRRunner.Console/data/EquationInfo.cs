namespace SCPRRunner.Console.Gridsearch.data {
  public class EquationInfo {
    public string AllowedInputs { get; set; }
    public string EquationName { get; set; }
    public string DescriptiveName { get; set; }
    public string TargetVariable { get; set; }
    public double TrainTestSplit { get; set; }
    public int[] Degrees { get; set; }
    public double[] Lambdas { get; set; }
    public double[] Alphas { get; set; }
    public int[] MaxInteractions { get; set; }
    public List<ConstraintInfo> Constraints { get; set; }
    public List<VariableInfo> Variables { get; set; }
  }
  public class ConstraintInfo {
    public string name { get; set; }
    public int order_derivative { get; set; }
    public string monotonicity { get; set; }
    public string derivative { get; set; }
    public string derivative_lambda { get; set; }
  }
  public class VariableInfo {
    public string name { get; set; }
    public double low { get; set; }
    public double high { get; set; }
  }
}
