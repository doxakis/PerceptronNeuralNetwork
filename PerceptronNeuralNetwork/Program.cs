using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronNeuralNetwork
{
	public class DataSet
	{
		public double[] Values { get; set; }
		public double[] Targets { get; set; }
	}

	class Program
	{
		static void Main(string[] args)
		{
			Stopwatch stopwatch = new Stopwatch();
			stopwatch.Start();
			
			// Csv file.
			var fileName = "iris.csv";

			// sepal_length,sepal_width,petal_length,petal_width
			int numberOfValues = 4; // Fields in csv file.

			// is_setosa,is_versicolor,is_virginica (can be: 0 = false or 1 = true)
			int numberOfTargets = 3; // Fields in csv file.

			PerceptronNetwork network = new PerceptronNetwork(
				inputSize: numberOfValues,
				hiddenSize: 3,
				outputSize: numberOfTargets,
				learnRate: 0.4,
				momentum: 0.95);

			List<DataSet> fullDataset = LoadCsv(fileName, numberOfValues, numberOfTargets);
			Shuffle(fullDataset);

			int validationProportion = 15; // %
			int crossValidationProportion = 15; // %

			List<DataSet> validationDataSet = fullDataset
				.Take(fullDataset.Count * validationProportion / 100)
				.ToList();

			List<DataSet> crossValidationDataSet = fullDataset
				.Skip(fullDataset.Count * validationProportion / 100)
				.Take(fullDataset.Count * crossValidationProportion / 100)
				.ToList();

			List<DataSet> trainDataset = fullDataset
				.Skip(fullDataset.Count *
					(100 - validationProportion - crossValidationProportion) / 100)
				.ToList();

			Console.WriteLine("Training started...\n");

			int iterations = 10;

			for (int i = 0; i < iterations; i++)
			{
				network.Train(
					values: trainDataset.Select(m => m.Values).ToArray(),
					targets: trainDataset.Select(m => m.Targets).ToArray(),
					numEpochs: 100);

				var errors = new List<double>();
				foreach (var dataSet in trainDataset)
				{
					network.Compute(dataSet.Values);
					errors.Add(network.CalculateError(dataSet.Targets));
				}
				var error = errors.Average();
				Console.WriteLine("Iteration #" + (i + 1) + ", cost = " + error);
			}

			Console.WriteLine("\nTraining completed.\n");

			Console.WriteLine("Testing network with train dataset:\n");
			TestNetwork(network, trainDataset);

			Console.WriteLine("Testing network with validation dataset:\n");
			TestNetwork(network, validationDataSet);

			Console.WriteLine("Testing network with cross validation dataset:\n");
			TestNetwork(network, crossValidationDataSet);

			stopwatch.Stop();
			Console.WriteLine("Duration: " + stopwatch.Elapsed);
			Console.WriteLine("Press any key to continue...");
			Console.ReadLine();
		}

		private static void TestNetwork(
			PerceptronNetwork network,
			List<DataSet> validationDataSet)
		{
			int nbGood = 0;
			int nbBad = 0;

			foreach (var dataSet in validationDataSet)
			{
				Console.Write("Input: " +
					string.Join(" ", dataSet.Values
						.Select(m => string.Format("{0:0.0}", m))));

				var result = network.Compute(dataSet.Values.ToArray());

				Console.Write(" Expected: " +
					string.Join(" ", dataSet.Targets
						.Select(m => string.Format("{0:0.0}", m))));

				Console.Write(" Computed: " +
					string.Join(" ", result
						.Select(m => string.Format("{0:0.0}", m))));

				int indexMaxExpected = !dataSet.Values.Any()
					? -1
					: dataSet.Targets
						.Select((value, index) => new { Value = value, Index = index })
						.Aggregate((a, b) => (a.Value > b.Value) ? a : b)
						.Index;

				int indexMaxComputed = !result.Any()
					? -1
					: result
						.Select((value, index) => new { Value = value, Index = index })
						.Aggregate((a, b) => (a.Value > b.Value) ? a : b)
						.Index;

				if (indexMaxComputed == indexMaxExpected)
				{
					Console.WriteLine(" GOOD RESULT");
					nbGood++;
				}
				else
				{
					Console.WriteLine(" BAD RESULT");
					nbBad++;
				}
			}

			Console.WriteLine();
			Console.WriteLine("# Good : " + nbGood);
			Console.WriteLine("# Bad  : " + nbBad);
			Console.WriteLine("% Good : " +
				string.Format("{0:0.0}", 100.0 * nbGood / (nbGood + nbBad)));
			Console.WriteLine();
		}

		private static List<DataSet> LoadCsv(
			string fileName,
			int numberOfValues,
			int numberOfTargets)
		{
			List<DataSet> myDataSet = new List<DataSet>();
			
			var lines = File.ReadLines(fileName)
				.Skip(1) /* Skip header. */;
			
			foreach (var line in lines)
			{
				var values = line.Split(',')
					.Take(numberOfValues)
					.Select(m => double.Parse(m, CultureInfo.InvariantCulture))
					.ToArray();

				var targets = line.Split(',')
					.Skip(numberOfValues)
					.Take(numberOfTargets)
					.Select(m => double.Parse(m, CultureInfo.InvariantCulture))
					.ToArray();

				myDataSet.Add(new DataSet { Values = values, Targets = targets });
			}
			
			return myDataSet;
		}

		public static void Shuffle<E>(IList<E> list)
		{
			Random rng = new Random();
			int n = list.Count;
			while (n > 1)
			{
				n--;
				int k = rng.Next(n + 1);
				E value = list[k];
				list[k] = list[n];
				list[n] = value;
			}
		}
	}
}
