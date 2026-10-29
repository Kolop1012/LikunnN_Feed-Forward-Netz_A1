package NeuralNetwork;

import NeuralNetwork.ActivationFunction;
import NeuralNetwork.CostFunction;
import NeuralNetwork.FFNeuralNetwork;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BackpropagationTest {

    @Test
    void testBackpropagationUpdatesWeightsAndReducesLoss() {
        int[] topology = new int[]{2, 3, 1};
        FFNeuralNetwork net = new FFNeuralNetwork(topology, ActivationFunction.SIGMOID, 123L);

        double[] inputs = new double[]{0.5, -0.5};
        double[] targets = new double[]{1.0};

        double[] initialOutput = net.feedForward(inputs);
        double initialLoss = CostFunction.HalfSquaredError.calculate(initialOutput, targets);

        double learningRate = 0.5;
        net.train(learningRate, targets);

        double[] newOutput = net.feedForward(inputs);
        double newLoss = CostFunction.HalfSquaredError.calculate(newOutput, targets);

        assertTrue(newLoss < initialLoss,
                String.format("Der Loss sollte sich durch Backpropagation verringern! Alt: %.6f, Neu: %.6f", initialLoss, newLoss));
    }

    @Test
    void testTrainThrowsExceptionWithoutPriorFeedForward() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{2, 2, 1}, ActivationFunction.SIGMOID, 42L);

        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            net.train(0.1, new double[]{1.0});
        });

        assertEquals("Vor jedem Lernschritt feedForward mit dem aktuellen Muster aufrufen.", exception.getMessage());
    }
}
