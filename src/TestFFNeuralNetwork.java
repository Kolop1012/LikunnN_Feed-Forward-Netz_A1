import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestFFNeuralNetwork {



    @BeforeEach
    void setup(){

    }

    @Test
    public void einfacherTest() {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{1, 0, 1}, x->x);
        double[] inputs = { 1 };
        double[] outputs = net.feedForward(inputs);
        assertEquals(2,outputs[0]);
    }
}
