package NeuralNetwork;

import NeuralNetwork.model.ActivationFunction;
import NeuralNetwork.model.FFNeuralNetwork;
import NeuralNetwork.train.Trainer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-End-Tests: Netz muss XOR in allen drei Modi lösen.
 */
class IntegrationTest {
    
    @ParameterizedTest(name = "XOR mit Modus {0} in {1} Epochen")
    @CsvSource({ "sgd,      5000", "momentum, 3000", "adaptive, 3000" })
    void xorConverges(String mode, int epochs) {
        double[][] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] targets = { { 0 }, { 1 }, { 1 }, { 0 } };
        
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 2, 4, 1 }, ActivationFunction.SIGMOID, 42L);
        Trainer.TrainResult r = new Trainer(net, mode, 1.0).train(inputs, targets, epochs);
        
        assertTrue(r.finalLoss() < 0.01,
                "Modus " + mode + ": Loss " + r.finalLoss() + " > 0.01 nach " + epochs + " Epochen");
        
        // Alle Muster korrekt klassifiziert
        for (int i = 0; i < inputs.length; i++) {
            double out = net.feedForward(inputs[i])[0];
            int predicted = out >= 0.5 ? 1 : 0;
            assertEquals((int) targets[i][0], predicted,
                    "Muster " + i + " (Eingabe " + java.util.Arrays.toString(inputs[i]) + ", Ausgabe " + out + ")");
        }
    }
}
