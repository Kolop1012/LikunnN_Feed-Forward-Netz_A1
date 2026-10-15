package NeuralNetwork;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestFFNeuralNetwork {
    
    ActivationFunction Linear = ActivationFunction.LINEAR;
    
    @Test
    public void einfacher1Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 1, 1 }, Linear);
        double[] inputs = { 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(2, outputs[0]);
    }
    
    @Test
    public void einfacher2Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 3, 1, 1, 1 }, Linear);
        double[] inputs = { 1, 1, 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(6, outputs[0]);
    }
    
    @Test
    public void einfacher3Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 5, 3, 2, 3 }, Linear);
        double[] inputs = { -1, 0, 1, 2, -3 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(3, outputs[0]);
        assertEquals(3, outputs[1]);
        assertEquals(3, outputs[2]);
    }
    
    @Test
    public void einfacher4Test() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 }, Linear);
        double[] inputs = { 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(9864101, outputs[0]);
    }

    @Test
    void testTrainingReducesLoss() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{2, 3, 1}, ActivationFunction.SIGMOID);
        double[] inputs = {1.0, 0.0};
        double[] target = {1.0}; // Der gewünschte Zielwert

        // 1. Initialen Loss (Fehler) vor dem Training berechnen
        double[] initialOutputs = net.feedForward(inputs);
        double initialLoss = CostFunction.MeanSquaredError.calculate(initialOutputs, target);

        // 2. Das Netzwerk für 150 Epochen trainieren
        double learningRate = 0.5;
        for (int epoch = 0; epoch < 150; epoch++) {
            // WICHTIG: Erst forward aufrufen, damit Layer die Zwischenwerte (lastInputs, lastRawInputs) cachen kann!
            net.feedForward(inputs);
            net.train(learningRate, target);
        }

        // 3. Finalen Loss nach dem Training berechnen
        double[] finalOutputs = net.feedForward(inputs);
        double finalLoss = CostFunction.MeanSquaredError.calculate(finalOutputs, target);

        // 4. Prüfen, ob der Fehler kleiner geworden ist
        assertTrue(finalLoss < initialLoss,
                "Der Fehler (Loss) sollte nach dem Training kleiner sein! " +
                        "Vorher: " + initialLoss + ", Nachher: " + finalLoss);
    }
}
