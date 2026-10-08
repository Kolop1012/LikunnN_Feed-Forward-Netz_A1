package NeuralNetwork;

import NeuralNetwork.ActivationFunction;
import NeuralNetwork.FFNeuralNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestFFNeuralNetwork {

    @Test
    public void einfacher1Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 1, 1 }, ActivationFunction.LINEAR);
        double[] inputs = { 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(2, outputs[0]);
    }

    @Test
    public void einfacher2Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 3, 1, 1, 1 }, ActivationFunction.LINEAR);
        double[] inputs = { 1, 1, 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(6, outputs[0]);
    }

    @Test
    public void einfacher3Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 5, 3, 2, 3 }, ActivationFunction.LINEAR);
        double[] inputs = { -1, 0, 1, 2, -3 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(3, outputs[0]);
        assertEquals(3, outputs[1]);
        assertEquals(3, outputs[2]);
    }

    @Test
    public void einfacher4Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 }, ActivationFunction.LINEAR);
        double[] inputs = { 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(9864101, outputs[0]);
    }
}
