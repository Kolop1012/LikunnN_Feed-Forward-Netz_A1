public class App {
    public static void main(String[] args) {
        // Erstellt ein Netz: 3 Input, 4 Hidden, 2 Output
        FFNeuralNetwork net = new FFNeuralNetwork(new int[] { 3, 4, 2 });

        double[] inputs = { 1.0, 0.5, -1.5 };
        double[] outputs = net.feedForward(inputs);

        System.out.println("Output 1: " + outputs[0]);
        System.out.println("Output 2: " + outputs[1]);
    }
}
