using Laughing_enigmaML.Model;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;

namespace laughing_enigma
{
    class Program
    {
        static void Main(string[] args)
        {
            RunTest();
        }

        private static void Predictor(string predictors, out int predicted, out double probability)
        {
            var mlContext = new MLContext();

            const string modelPath = @"C:\Users\keith\AppData\Local\Temp\MLVSTools\laughing-enigmaML\laughing-enigmaML.Model\MLModel.zip";

            var mlModel = mlContext.Model.Load(modelPath, out _);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            var modelInput = new ModelInput();

            var fields = predictors.Split(',');

            modelInput.BoxRatio = Convert.ToSingle(fields[0]);
            modelInput.Thrust = Convert.ToSingle(fields[1]);
            modelInput.Velocity = Convert.ToSingle(fields[2]);
            modelInput.OnBalRun = Convert.ToSingle(fields[3]);
            modelInput.VwapGain = Convert.ToSingle(fields[4]);

            var prediction = predEngine.Predict(modelInput);

            //predicted = prediction.Prediction ? 1 : 0;
            predicted = Convert.ToBoolean(prediction.Score) ? 1 : 0;

            probability = 0.0;
        }

        private static void RunTest()
        {
            int predicted, false_positive, true_negative, false_negative;

            var true_positive = false_positive = true_negative = false_negative = 0;

            double goal_met = 0.0, goal_failed = 0.0, probability;

            var data = new List<string>();

            var url = "Data\rockettest.csv";

            var req = (HttpWebRequest)WebRequest.Create(url);

            var resp = (HttpWebResponse)req.GetResponse();

            var reader = new StreamReader(resp.GetResponseStream());

            reader.ReadLine(); // Ignore the header.

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();

                data.Add(line);
            }

            var loop = 0;

            foreach (var predictors in data)
            {
                Predictor(predictors, out predicted, out probability);

                var fields = predictors.Split(',');

                var actual = Convert.ToInt32(fields[5]);

                if (actual == 1) goal_met++; else goal_failed++;

                if (predicted == 1 && actual == 1)
                {
                    true_positive += 1;
                }

                if (predicted == 1 && actual == 0)
                {
                    false_positive += 1; // False Positive. The model predicted a gain, but we got a loss.
                }

                if (predicted == 0 && actual == 0)
                {
                    true_negative += 1; // True Negative. The model predicted a loss, and we got a loss.
                }

                if (predicted == 0 && actual == 1)
                {
                    false_negative += 1; // False Negative. The model predicted a loss, but and we got a gain.
                }

                Console.Write(++loop + ": " + true_positive + ", " + false_positive + ", " + false_negative + ", " + true_negative + ", " + false_negative + ", " + true_negative + "\r");
            }

            double recall, f1score;

            var precision = recall = f1score = 0.0;

            var denom = Convert.ToDouble(true_positive + false_positive);

            if (denom > 0.0) precision = true_positive / denom;

            denom = Convert.ToDouble(true_positive + false_negative);

            if (denom > 0.0) recall = true_positive / denom;

            if (precision + recall > 0.0)
            {
                f1score = 2.0 * (precision * recall) / (precision + recall);
            }

            precision = Math.Round(precision, 4);

            recall = Math.Round(recall, 4);

            f1score = Math.Round(f1score, 4);

            Console.WriteLine("No. of True Positive...." + true_positive + "/" + goal_met);
            Console.WriteLine("No. of False Positive..." + false_positive);
            Console.WriteLine("No. of False Negative..." + false_negative);
            Console.WriteLine("No. of True Negative...." + true_negative + "/" + goal_failed);
            Console.WriteLine("Precision..............." + precision);
            Console.WriteLine("Recall.................." + recall);
            Console.WriteLine("F1 Score................" + f1score);

            Console.WriteLine("End of process, press any key to finish.");

            Console.ReadKey();
        }
    }
}
