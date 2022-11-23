namespace SCPRRunner.Console.Gridsearch.data {
  public class SCPRResult : IComparable {
    public string EquationName { get; set; }
    public string VariedVariable { get; set; }
    public string DataFilePath { get; set; }

    public double Degree { get; set; }
    public double Lambda { get; set; }
    public double Alpha { get; set; }
    public double MaxInteractions { get; set; }

    public long Runtime { get; set; } = 0;
    public bool Successful { get; set; } = false;

    public double RMSE_Training { get; set; }
    public double RMSE_Test { get; set; }
    public double RMSE_Full { get; set; }

    public int CompareTo(object? obj) {
      if (!(obj is SCPRResult)) return -1;
      var other = obj as SCPRResult;
      if (other == null) return -1;

      int comp;
      if ((comp = other.EquationName.CompareTo(EquationName)) != 0)
        return comp;
      if ((comp = other.VariedVariable.CompareTo(VariedVariable)) != 0)
        return comp;
      if ((comp = other.DataFilePath.CompareTo(DataFilePath)) != 0)
        return comp;
      if ((comp = other.Degree.CompareTo(Degree)) != 0)
        return comp;
      if ((comp = other.Lambda.CompareTo(Lambda)) != 0)
        return comp;
      if ((comp = other.Alpha.CompareTo(Alpha)) != 0)
        return comp;
      return 0;
    }
  }
}
