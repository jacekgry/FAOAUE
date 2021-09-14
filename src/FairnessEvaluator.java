package moa.evaluation;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.Utils;

import java.util.Arrays;

public class FairnessEvaluator extends BasicClassificationPerformanceEvaluator {

    public IntOption positiveClass = new IntOption("positiveClass", 'P', "The index of positive class", 0);
    public IntOption sensitiveAttribute = new IntOption("sensitiveAttribute", 's', "The index of the sensitive attribute", 0);
    public IntOption protectedValue = new IntOption("protectedValue", 'v', "The index of the protected value", 0);

    private static final double L = 0.001;

    private int numOfProtected;
    private int numOfNotProtected;
    private int numOfProtectedPredictedPositive;
    private int numOfNotProtectedPredictedPositive;

    private int numOfPositiveProtected;
    private int numOfPositiveNotProtected;
    private int numOfPositiveProtectedClassifiedPositive;
    private int numOfPositiveNotProtectedClassifiedPositive;

    @Override
    public Measurement[] getPerformanceMeasurements() {
        Measurement[] measurements = super.getPerformanceMeasurements();
        measurements = Arrays.copyOf(measurements, measurements.length + 2);

        measurements[measurements.length - 2] = new Measurement("Cumulative Statistical Parity", getCumulativeStatisticalParity());
        measurements[measurements.length - 1] = new Measurement("Cumulative Equal Opportunity", getCumulativeEqualOpportunity());
        return measurements;
    }

    @Override
    public void addResult(Example<Instance> example, double[] classVotes) {
        super.addResult(example, classVotes);

        boolean isPositive = (int) (example.getData().classValue()) == positiveClass.getValue();
        boolean isProtected = (int) (example.getData().value(sensitiveAttribute.getValue())) == protectedValue.getValue();
        boolean isPredictedPositive = Utils.maxIndex(classVotes) == positiveClass.getValue();

        if (isProtected) {
            numOfProtected++;
            if (isPositive) {
                numOfPositiveProtected++;
                if (isPredictedPositive) {
                    numOfPositiveProtectedClassifiedPositive++;
                }
            }
            if (isPredictedPositive) {
                numOfProtectedPredictedPositive++;
            }
        } else {
            numOfNotProtected++;
            if (isPositive) {
                numOfPositiveNotProtected++;
                if (isPredictedPositive) {
                    numOfPositiveNotProtectedClassifiedPositive++;
                }
            }
            if (isPredictedPositive) {
                numOfNotProtectedPredictedPositive++;
            }
        }
    }

    private double getCumulativeStatisticalParity() {
        return (double) numOfNotProtectedPredictedPositive / (numOfNotProtected + L) - (double) numOfProtectedPredictedPositive / (numOfProtected + L);
    }

    private double getCumulativeEqualOpportunity() {
        return (double) numOfPositiveNotProtectedClassifiedPositive / (numOfPositiveNotProtected + L) - (double) numOfPositiveProtectedClassifiedPositive / (numOfPositiveProtected + L);
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == FairnessEvaluator.class) {
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        } else {
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
        }
    }

}
