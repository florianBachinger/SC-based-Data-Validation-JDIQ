using System.Globalization;
using CsvHelper;
using Newtonsoft.Json;
using SCPRRunner.Console.Gridsearch.data;
using ShapeConstrainedPolynomialRegression;
using static ShapeConstrainedPolynomialRegression.Regression;
using ConstraintTuple = System.Tuple<string, int, double, int, double[], double[]>;

int argumentIndex = 0;
foreach(var argument in Environment.GetCommandLineArgs()) {
  Console.WriteLine($"argument[{argumentIndex}]: {argument}");
  argumentIndex++;
}

Console.WriteLine("Expected Argument Order: ");
Console.WriteLine("argument[1]: resulting csv file location");
Console.WriteLine("argument[2]: training data folder for individual models");
Console.WriteLine("argument[3]: resulting data folder for individual model reports");
Console.WriteLine("argument[4]: int degree of parallelism");

//prepare data sources and result path
string mlResultFile = Environment.GetCommandLineArgs()[1];
string dataFolder = Environment.GetCommandLineArgs()[2];
string resultFolder = Environment.GetCommandLineArgs()[3];
int degreeParallelism = int.Parse(Environment.GetCommandLineArgs()[4]);
var file_write_lock = new object();


//read previous results
List<SCPRResult> results;
using (var reader = new StreamReader(mlResultFile, new FileStreamOptions() { Access = FileAccess.ReadWrite, Mode = FileMode.OpenOrCreate }))
using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
  results = csv.GetRecords<SCPRResult>().ToList();
}

//calculate for each data file
Parallel.ForEach(Directory.GetFiles(dataFolder, "*.csv"),
  new ParallelOptions { MaxDegreeOfParallelism = degreeParallelism },
dataFile => {
  try {
    List<dynamic> records;

    var dataFile_FileName = Path.GetFileNameWithoutExtension(dataFile);


    using (var reader = new StreamReader(dataFile))
    using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
      records = csv.GetRecords<dynamic>().ToList();
    }

    var first_record = records.First();
    var equation_name = first_record.equation_name;

    var fileContent = File.ReadAllText(Path.Combine(dataFolder, $"{equation_name}.json"));

    var equation_parameters = JsonConvert.DeserializeObject<EquationInfo>(fileContent);
    var variables = equation_parameters.Variables.ToArray();
    var target = equation_parameters.TargetVariable;
    var train_test_split = equation_parameters.TrainTestSplit;

    var countAll = records.Count();
    var front = (int)(countAll * train_test_split);
    var back = (int)(countAll - front);

    var training_records = records.Take(front);
    var test_records = records.TakeLast(back);
    var variable_name = "";

    string[] inputs = variables.Select(x => x.name).ToArray(); ;
    if (equation_parameters.AllowedInputs == "only_varied") {
      variable_name = first_record.varied_variable_name;
      inputs = new string[] { variable_name };
    } else {
      inputs = variables.Select(x => x.name).ToArray();
    }
    var constraints = CreateConstraint(equation_parameters, inputs).ToArray();


    ExtractScaledData(records, variables, inputs, target, out double[,] X_full, out double[] y_full);
    ExtractScaledData(training_records, variables, inputs, target, out double[,] X_train, out double[] y_train);
    ExtractScaledData(test_records, variables, inputs, target, out double[,] X_test, out double[] y_test);


    foreach (var degree in equation_parameters.Degrees) {
      foreach (var interactions in equation_parameters.MaxInteractions) {
        foreach (var lambda in equation_parameters.Lambdas) {
          foreach (var alpha in equation_parameters.Alphas) {



            var predictionResultFileName = $"{dataFile_FileName}_d{degree}_i{interactions}_l{lambda}_a{alpha}.csv";
            var result_path = Path.Combine(resultFolder, predictionResultFileName);

            var result_record = new SCPRResult() {
              EquationName = equation_name,
              DataFilePath = dataFile,
              VariedVariable = variable_name,
              Degree = degree,
              Lambda = lambda,
              Alpha = alpha,
              MaxInteractions = interactions,
              Runtime = 0,
              Successful = false,
              RMSE_Full = -1,
              RMSE_Test = -1,
              RMSE_Training = -1
            };


            //not calculated yet
            if (!results.Any(x => x.CompareTo(result_record) == 0)) {

              var tokenSource = new CancellationTokenSource();
              var timer = new System.Timers.Timer(TimeSpan.FromMinutes(2.5).TotalMilliseconds); // fire every 1 second
              timer.Elapsed += delegate {
                Console.WriteLine($"file: {dataFile} degree: {degree} interactions:{interactions} lambda:{lambda} alpha: {alpha} TIMER ELAPSED");

                tokenSource.Cancel();
                WriteAppend(mlResultFile, result_record);

              };

              var watch = new System.Diagnostics.Stopwatch();
              watch.Start();

              if (TrainModel(X: X_train, y: y_train, inputs: inputs, target: target,
                            degree: degree, alpha: alpha, lambda: lambda,
                            max_interactions: interactions, constraints: constraints, cancellationToken: tokenSource.Token, out Polynomial polynomial,
                            out Result regressionResult)) {

                watch.Stop();
                Console.WriteLine($"Execution Time: {watch.ElapsedMilliseconds} ms");

                result_record.Runtime = watch.ElapsedMilliseconds;
                result_record.Successful = true;

                var y_predicted = CalculatePrediction(polynomial, X_full);
                result_record.RMSE_Test = CalculateRMSE(y_test, CalculatePrediction(polynomial, X_test));
                result_record.RMSE_Training = CalculateRMSE(y_train, CalculatePrediction(polynomial, X_train));
                result_record.RMSE_Full = CalculateRMSE(y_full, y_predicted);
                for (int i = 0; i < records.Count(); i++) {
                  records.ElementAt(i).Predicted = y_predicted[i];
                }

                Console.WriteLine($"file: {dataFile} degree: {degree} interactions:{interactions} lambda:{lambda} alpha: {alpha} RMSE_Training: {result_record.RMSE_Training} RMSE_Test: {result_record.RMSE_Test} RMSE_Full:{result_record.RMSE_Full}");

                if (!Directory.Exists(Path.GetDirectoryName(result_path))) Directory.CreateDirectory(Path.GetDirectoryName(result_path));
                if (!File.Exists(result_path))
                  using (File.Create(result_path)) { }

                using (var writer = new StreamWriter(result_path, append: false))
                using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture)) {
                  csv.WriteRecords(records);
                  csv.Flush();
                  writer.Flush();
                }
              }
              WriteAppend(mlResultFile, result_record);
            }
          }
        }
      }
    }
  }
  catch (Exception ex) {
    Console.WriteLine(ex.Message);
  }
});


void ExtractScaledData(IEnumerable<dynamic> test_records, VariableInfo[] variables,string[] allowed_inputs, string target, out double[,] X, out double[] y) {
  X = new double[test_records.Count(), allowed_inputs.Count()];
  y = new double[test_records.Count()];
  for (int i = 0; i < test_records.Count(); i++) {
    var row_record = (IDictionary<string, dynamic>)test_records.ElementAt(i);
    for (int j = 0; j < variables.Count(); j++) {
      var var_index = Array.IndexOf(allowed_inputs, variables[j].name);
      if (var_index>=0) {
        var val = double.Parse(row_record[variables[j].name], CultureInfo.InvariantCulture);
        X[i, var_index] = Scale(val, variables[j].low, variables[j].high);
      }
    }
    y[i] = double.Parse(row_record[target], CultureInfo.InvariantCulture);
  }
}

double[] CalculatePrediction(Polynomial optimizedModel, double[,] x) {
  return Regression.Evaluate(optimizedModel, x);
}

bool TrainModel(double[,] X, double[] y, string[] inputs, string target, int degree, double alpha, double lambda,
                int max_interactions, Tuple<string, int, double, int, double[], double[]>[] constraints, CancellationToken cancellationToken,
                out Polynomial optimizedModel, out Result regressionResult) {

  optimizedModel = default;
  regressionResult = default;
  try {
    optimizedModel = ShapeConstrainedPolynomialRegression.Regression.Run(variableNames: inputs,
                                    X,
                                    y,
                                    degree: degree,
                                    maxVarsInInteraction: max_interactions,
                                    constraints: constraints,
                                    lambda: lambda,
                                    alpha: alpha,
                                    out regressionResult,
                                    scaleFeatures: true,
                                    approximation: "SDP",
                                    positivstellensatzDegree: -1,
                                    cancellationToken: cancellationToken);
  }
  catch (Exception e) {
    Console.WriteLine(e.ToString());
  }
  if (optimizedModel != null) {
    return true;
  }
  return false;
}



void WriteAppend<T>(string filePath, T record) {
  // This text is added only once to the file.
  if (!File.Exists(filePath)) {
    File.Create(filePath);
  } else {
    // Create a file to write to.
    lock (file_write_lock) {
      using (StreamWriter sw = new StreamWriter(filePath, append: true)) {
        using (var csv = new CsvWriter(sw, CultureInfo.InvariantCulture)) {
          if ((new FileInfo(filePath)).Length == 0) {
            csv.WriteHeader<T>();
            csv.NextRecord();
          }
          csv.WriteRecord(record);
          csv.NextRecord();
          sw.Flush();
        }
        sw.Close();
      }
    }
  }
}

T[,] Shuffle2D<T>(Random rng, T[,] old) {
  T[,] array = new T[old.GetLength(0), old.GetLength(1)];
  Array.Copy(old, array, old.Length);
  int n = array.GetLength(0);
  while (n > 1) {
    int k = rng.Next(n--);
    for (int j = 0; j < old.GetLength(1); j++) {
      T temp = array[n, j];
      array[n, j] = array[k, j];
      array[k, j] = temp;
    }
  }
  return array;
}

T[] Shuffle1D<T>(Random rng, T[] old) {
  T[] array = new T[old.Length];
  Array.Copy(old, array, old.Length);
  int n = array.Length;
  while (n > 1) {
    int k = rng.Next(n--);
    T temp = array[n];
    array[n] = array[k];
    array[k] = temp;
  }
  return array;
}

double CalculateRMSE(double[] y_target, double[] y_predicted) {
  if (y_target.Length != y_predicted.Length)
    throw new ArgumentException($"Length of the two arrays {nameof(y_target)} and {nameof(y_predicted)} must match.");
  var sumSqRes = 0.0;
  for (int i = 0; i < y_predicted.Length; i++) sumSqRes += (y_target[i] - y_predicted[i]) * (y_target[i] - y_predicted[i]);
  return Math.Sqrt(sumSqRes / y_predicted.Length);
}

double Scale(double val, double minimum, double maximum) {

  double range = maximum - minimum;
  double multiplier = 2.0 / range;
  double addend = -1 - multiplier * minimum;

  return val * multiplier + addend;
}

IEnumerable<ConstraintTuple> CreateConstraint(EquationInfo equationInfo, string[] allowed_inputs) {
  var constraints = equationInfo.Constraints.Where(x => allowed_inputs.Contains(x.name)).ToArray();
  var inputVariables = equationInfo.Variables.Where(x => allowed_inputs.Contains(x.name)).Select(x => x.name).ToArray();
  var lb = inputVariables.Select(var => -1.0).ToArray();
  var ub = inputVariables.Select(var => 1.0).ToArray();

  foreach (var constraint in constraints) {
    var input_name = constraint.name;
    var order_derivative = constraint.order_derivative;

    if (constraint.monotonicity == "increasing")
      yield return Tuple.Create(input_name,
                    order_derivative,
                    0.0,                    // lowest function value
                    1,                      // highest function value
                    (double[])lb.Clone(),   // for the full input space
                    (double[])ub.Clone());  // for the full input space
    if (constraint.monotonicity == "decreasing")
      yield return Tuple.Create(input_name,
                    order_derivative,
                    0.0,                    // highest function value
                    -1,                     // lowest function value
                    (double[])lb.Clone(),   // for the full input space
                    (double[])ub.Clone());  // for the full input space
    if (constraint.monotonicity == "constant")
      yield return Tuple.Create(input_name,
                    order_derivative,
                    0.0,                    // highest function value
                    0,                      // lowest function value
                    (double[])lb.Clone(),   // for the full input space
                    (double[])ub.Clone());  // for the full input space

  }
}