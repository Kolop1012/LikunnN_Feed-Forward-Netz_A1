package NeuralNetwork;

/** Reduziert bei einem Fehlerplateau die Lernrate bis zur angegebenen Untergrenze. */
public class LearningRateScheduler {
    private final int patience;
    private final double factor;
    private final double minLearningRate;
    private final double minImprovement;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private int epochsWithoutImprovement;

    public LearningRateScheduler(int patience, double factor, double minLearningRate, double minImprovement) {
        if (patience < 1 || !Double.isFinite(factor) || factor <= 0.0 || factor >= 1.0
                || !Double.isFinite(minLearningRate) || minLearningRate <= 0.0
                || !Double.isFinite(minImprovement) || minImprovement < 0.0) {
            throw new IllegalArgumentException("Ungültige Parameter für die Lernratenanpassung.");
        }
        this.patience = patience;
        this.factor = factor;
        this.minLearningRate = minLearningRate;
        this.minImprovement = minImprovement;
    }

    /** Einmal nach jeder Epoche mit dem Fehler bei unveränderten Gewichten aufrufen. */
    public double nextRate(double currentRate, double loss) {
        if (!Double.isFinite(currentRate) || currentRate < minLearningRate
                || !Double.isFinite(loss) || loss < 0.0) {
            throw new IllegalArgumentException("Lernrate und Fehler müssen gültige endliche Werte sein.");
        }
        if (loss < bestLoss - minImprovement) {
            bestLoss = loss;
            epochsWithoutImprovement = 0;
        } else if (++epochsWithoutImprovement >= patience) {
            epochsWithoutImprovement = 0;
            return Math.max(minLearningRate, currentRate * factor);
        }
        return currentRate;
    }
}
