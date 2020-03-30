
using Microsoft.ML;
using System.Linq;
using static System.Console;


namespace LearnML
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                MLContext context = new MLContext();
                //Load Data 
                var trainData = context.Data.LoadFromTextFile<SalaryData>("SalaryData.csv", hasHeader: true, separatorChar: ',');
                //var testTrainSplit = context.Data.TrainTestSplit(trainData, testFraction: 0.1);
                //Build Model
                var pipeline = context.Transforms.Concatenate("Features", "YearExperience")
                    .Append(context.Regression.Trainers.LbfgsPoissonRegression());
                var model = pipeline.Fit(trainData);
                //Evaluate
                var prediction = model.Transform(trainData);

                var metrics = context.Regression.Evaluate(prediction);

                WriteLine($"RSquared : {metrics.RSquared}");
                WriteLine($"MeanAbsoluteError : {metrics.MeanAbsoluteError}");
                WriteLine($"MeanSquaredError : {metrics.MeanSquaredError}");
                WriteLine("========================*=======================");
                //Prediction
                for (int i = 3;i< 10; i++)
                {
                    var newData = new SalaryData()
                    {
                        YearExperience = i + 0.3f
                    };
                    var predictedFunction = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);
                    var testPredict = predictedFunction.Predict(newData);

                    WriteLine($"Year Experience => {newData.YearExperience} his salary will be (Mechine Prediction) : {testPredict.PredictionSalary}");
                }
                ReadKey();
            }
            catch (System.Exception ex)
            {
                Write(ex);
                ReadLine();
            }

         



        }
    }
}
