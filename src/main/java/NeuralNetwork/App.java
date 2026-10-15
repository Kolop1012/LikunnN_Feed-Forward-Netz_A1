package NeuralNetwork;

public class App {
    static void main(String[] args) {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{3, 1, 2}, ActivationFunction.LINEAR);

        double[] inputs = {1, 0, 1};
        double[] outputs = net.feedForward(inputs);
        double[] targets = new double[0]; // Erwartete werte
        double loss = CostFunction.MeanSquaredError.calculate(outputs, targets);

        IO.println("Output 1: " + outputs[0]);
        IO.println("Output 2: " + outputs[1]);
    }
}
