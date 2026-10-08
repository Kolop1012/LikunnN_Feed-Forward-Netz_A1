package NeuralNetwork;

public class App {
    public static void main(String[] args) {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 3, 1, 2 }, ActivationFunction.LINEAR);

        double[] inputs = { 1, 0, 1 };
        double[] outputs = net.feedForward(inputs);

        System.out.println("Output 1: " + outputs[0]);
        System.out.println("Output 2: " + outputs[1]);
    }
}
