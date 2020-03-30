using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace LearnML
{
    public  class SalaryData
    {

        [LoadColumn(0)]
        public float YearExperience;

        [LoadColumn(1), ColumnName("Label")]
        public float Salary;
    }
}
