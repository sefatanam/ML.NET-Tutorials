using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace LearnML
{
     public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictionSalary { get; set; }
    }
}
