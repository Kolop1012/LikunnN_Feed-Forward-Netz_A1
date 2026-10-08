public class App {
    public static void main(String[] args) {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 3, 1, 2 }, x->x);

        double[] inputs = { 1, 0, 1 };
        double[] outputs = net.feedForward(inputs);

        System.out.println("Output 1: " + outputs[0]);
        System.out.println("Output 2: " + outputs[1]);
    }

    // Aktivierungsfunktion
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
