using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronNeuralNetwork
{
	public class PerceptronNetwork
	{
		public int InputSize { get; set; }
		public int HiddenSize { get; set; }
		public int OutputSize { get; set; }
		public double LearnRate { get; set; }
		public double Momentum { get; set; }

		// Hidden layer
		public double[] HiddenBias { get; set; }
		public double[] HiddenBiasDelta { get; set; }
		public double[] HiddenGradient { get; set; }
		public double[] HiddenValue { get; set; }
		public double[] HiddenWeight { get; set; }
		public double[] HiddenWeightDelta { get; set; }

		// Output layer
		public double[] OutputBias { get; set; }
		public double[] OutputBiasDelta { get; set; }
		public double[] OutputGradient { get; set; }
		public double[] OutputValue { get; set; }
		public double[] OutputWeight { get; set; }
		public double[] OutputWeightDelta { get; set; }

		public PerceptronNetwork(
			int inputSize,
			int hiddenSize,
			int outputSize,
			double learnRate,
			double momentum)
		{
			InputSize = inputSize;
			HiddenSize = hiddenSize;
			OutputSize = outputSize;
			LearnRate = learnRate;
			Momentum = momentum;

			HiddenBias = new double[hiddenSize];
			for (int i = 0; i < HiddenBias.Length; i++)
			{
				HiddenBias[i] = GetRandom();
			}
			HiddenBiasDelta = new double[hiddenSize];
			HiddenGradient = new double[hiddenSize];
			HiddenValue = new double[hiddenSize];
			HiddenWeight = new double[hiddenSize * inputSize];
			HiddenWeightDelta = new double[hiddenSize * inputSize];
			for (int i = 0; i < HiddenWeight.Length; i++)
			{
				HiddenWeight[i] = GetRandom();
			}

			OutputBias = new double[outputSize];
			for (int i = 0; i < OutputBias.Length; i++)
			{
				OutputBias[i] = GetRandom();
			}
			OutputBiasDelta = new double[outputSize];
			OutputGradient = new double[outputSize];
			OutputValue = new double[outputSize];
			OutputWeight = new double[outputSize * hiddenSize];
			OutputWeightDelta = new double[outputSize * hiddenSize];
			for (int i = 0; i < OutputWeight.Length; i++)
			{
				OutputWeight[i] = GetRandom();
			}
		}

		public void Train(double[][] values, double[][] targets, int numEpochs)
		{
			if (values.Length != targets.Length)
			{
				throw new ArgumentException("Values and targets are not the same length.");
			}

			for (var i = 0; i < numEpochs; i++)
			{
				for (int j = 0; j < values.Length; j++)
				{
					ForwardPropagate(values[j]);
					BackPropagate(values[j], targets[j]);
				}
			}
		}

		private void ForwardPropagate(double[] inputs)
		{
			// Hidden layer.
			for (int i = 0; i < HiddenSize; i++)
			{
				var x = HiddenBias[i];
				for (int j = 0; j < InputSize; j++)
				{
					int pos = j + i * InputSize;
					x += HiddenWeight[pos] * inputs[j];
				}
				HiddenValue[i] = SigmoidOutput(x);
			}

			// Output layer.
			for (int i = 0; i < OutputSize; i++)
			{
				var x = OutputBias[i];
				for (int j = 0; j < HiddenSize; j++)
				{
					int pos = j + i * HiddenSize;
					x += OutputWeight[pos] * HiddenValue[j];
				}
				OutputValue[i] = SigmoidOutput(x);
			}
		}

		private void BackPropagate(double[] inputs, double[] targets)
		{
			// Calculate gradient (output layer)
			for (int i = 0; i < OutputSize; i++)
			{
				OutputGradient[i] = (targets[i] - OutputValue[i]) * SigmoidDerivative(OutputValue[i]);
			}

			// Calculate gradient (hidden layer)
			for (int i = 0; i < HiddenSize; i++)
			{
				double sum = 0;
				for (int j = 0; j < OutputSize; j++)
				{
					sum += OutputGradient[j] * OutputWeight[i + j * HiddenSize];
				}
				HiddenGradient[i] = sum * SigmoidDerivative(HiddenValue[i]);
			}

			// Update weights (hidden layer)
			for (int i = 0; i < HiddenSize; i++)
			{
				var prevDelta = HiddenBiasDelta[i];
				HiddenBiasDelta[i] = LearnRate * HiddenGradient[i];
				HiddenBias[i] += HiddenBiasDelta[i] + Momentum * prevDelta;

				for (int j = 0; j < InputSize; j++)
				{
					int pos = j + i * InputSize;
					prevDelta = HiddenWeightDelta[pos];

					HiddenWeightDelta[pos] = LearnRate * HiddenGradient[i] * inputs[j];

					HiddenWeight[pos] += HiddenWeightDelta[pos] + Momentum * prevDelta;
				}
			}

			// Update weights (output layer)
			for (int i = 0; i < OutputSize; i++)
			{
				var prevDelta = OutputBiasDelta[i];
				OutputBiasDelta[i] = LearnRate * OutputGradient[i];
				OutputBias[i] += OutputBiasDelta[i] + Momentum * prevDelta;

				for (int j = 0; j < HiddenSize; j++)
				{
					int pos = j + i * HiddenSize;
					prevDelta = OutputWeightDelta[pos];
					OutputWeightDelta[pos] = LearnRate * OutputGradient[i] * HiddenValue[j];
					OutputWeight[pos] += OutputWeightDelta[pos] + Momentum * prevDelta;
				}
			}
		}

		public double[] Compute(double[] inputs)
		{
			ForwardPropagate(inputs);
			return OutputValue;
		}

		public double CalculateError(params double[] targets)
		{
			double error = 0;
			for (int i = 0; i < OutputSize; i++)
			{
				error += Math.Abs(targets[i] - OutputValue[i]);
			}
			return error;
		}

		public static double SigmoidOutput(double x)
		{
			return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
		}

		public static double SigmoidDerivative(double x)
		{
			return x * (1 - x);
		}

		private static readonly Random Random = new Random();

		public static double GetRandom()
		{
			return 2 * Random.NextDouble() - 1;
		}
	}
}
