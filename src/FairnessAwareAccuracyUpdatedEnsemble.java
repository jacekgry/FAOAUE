package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.core.Utils;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;

public class FairnessAwareAccuracyUpdatedEnsemble extends OnlineAccuracyUpdatedEnsemble {

    private static final double L = 0.001;

    public FloatOption discriminationToleranceThreshold = new FloatOption("threshold", 't', "The discrimination tolerance threshold", 0.001);
    public IntOption discriminationWindowSize = new IntOption(
      "discriminationWindowSize", 'W', "The discrimination windows size used to adjust decision boundary", 2000);

    public IntOption positiveClass = new IntOption("positiveClass", 'P', "The index of positive class", 0);
    public IntOption sensitiveAttribute = new IntOption("sensitiveAttribute", 's', "The index of the sensitive attribute", 0);
    public IntOption protectedValue = new IntOption("protectedValue", 'V', "The index of the protected value", 0);
    public StringOption goal = new StringOption("goal", 'g', "Fairness measure to optimize, either eqOpp or stPar", "eqOpp");

    private int numOfProtected;
    private int numOfNotProtected;
    private int numOfProtectedPredictedPositive;
    private int numOfNotProtectedPredictedPositive;

    private int numOfPositiveProtected;
    private int numOfPositiveNotProtected;
    private int numOfPositiveProtectedClassifiedPositive;
    private int numOfPositiveNotProtectedClassifiedPositive;

    private Deque<MisclassifiedExample> protectedMisclassifiedExamples = new ArrayDeque<>();

    private double decisionBoundaryForProtected = 0.5;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] votesForInstance = super.getVotesForInstance(inst);
        if (votesForInstance.length < 2) {
            return votesForInstance;
        }

        for (int i = 0; i < votesForInstance.length; i++) {
            if (Double.isInfinite(votesForInstance[i])) {
                votesForInstance[i] = 1.0;
            }
            if (Double.isNaN(votesForInstance[i])) {
                votesForInstance[i] = 0.0;
            }
        }
        Utils.normalize(votesForInstance);

        votesForInstance = adjustVotes(inst, votesForInstance);
        return votesForInstance;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        super.trainOnInstanceImpl(inst);

        if (!protectedMisclassifiedExamples.isEmpty() && protectedMisclassifiedExamples.getFirst().timestamp < processedInstances - discriminationWindowSize.getValue()) {
            protectedMisclassifiedExamples.removeFirst();
        }

        boolean isProtected = isInstanceProtected(inst);
        boolean isPositive = (int) (inst.classValue()) == positiveClass.getValue();

        double[] originalVotes = super.getVotesForInstance(inst);

        if (originalVotes.length < 2) {
            return;
        }

        double[] adjustedVotes = originalVotes.clone();

        for (int i = 0; i < adjustedVotes.length; i++) {
            if (Double.isInfinite(adjustedVotes[i])) {
                adjustedVotes[i] = 1.0;
            }
            if (Double.isNaN(adjustedVotes[i])) {
                adjustedVotes[i] = 0.0;
            }
        }

        Utils.normalize(adjustedVotes);
        adjustedVotes = adjustVotes(inst, adjustedVotes);

        boolean isPredictedPositive = Utils.maxIndex(adjustedVotes) == positiveClass.getValue();

        if (isProtected && isPositive && !isPredictedPositive) {
            protectedMisclassifiedExamples.add(new MisclassifiedExample(processedInstances, adjustedVotes[positiveClass.getValue()]));
        }

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

        if (goal.getValue().equals("eqOpp")) {
            double cumEqOp = getCumulativeEqualOpportunity();
            if (cumEqOp > discriminationToleranceThreshold.getValue()) {
                adjustBoundaryDecisionEqOp();
            }
            if (cumEqOp < -discriminationToleranceThreshold.getValue()) {
                decisionBoundaryForProtected = 0.5;
            }
        } else if (goal.getValue().equals("stPar")) {
            double cumStPar = getCumulativeStatisticalParity();
            if (cumStPar > discriminationToleranceThreshold.getValue()) {
                adjustBoundaryDecisionStPar();
            }
            if (cumStPar < -discriminationToleranceThreshold.getValue()) {
                decisionBoundaryForProtected = 0.5;
            }
        }

    }

    private void adjustBoundaryDecisionEqOp() {
        int numOfInstancesToMitigate = (int) Math.floor((double) numOfPositiveProtected * numOfPositiveNotProtectedClassifiedPositive / numOfPositiveNotProtected
                - numOfPositiveProtectedClassifiedPositive
        );
        Double[] confidences = protectedMisclassifiedExamples.stream().map(m -> m.confidence).toArray(Double[]::new);
        Arrays.sort(confidences, Collections.reverseOrder());
        if (numOfInstancesToMitigate > 0 && confidences.length > 0) {
            if (numOfInstancesToMitigate > confidences.length - 1) {
                decisionBoundaryForProtected = confidences[confidences.length - 1];
            } else {
                decisionBoundaryForProtected = confidences[numOfInstancesToMitigate];
            }
        }
    }

    private void adjustBoundaryDecisionStPar() {
        int numOfInstancesToMitigate = (int) Math.floor((double) numOfProtected * numOfNotProtectedPredictedPositive / numOfNotProtected
                - numOfProtectedPredictedPositive
        );
        Double[] confidences = protectedMisclassifiedExamples.stream().map(m -> m.confidence).toArray(Double[]::new);
        Arrays.sort(confidences, Collections.reverseOrder());
        if (numOfInstancesToMitigate > 0 && confidences.length > 0) {
            if (numOfInstancesToMitigate > confidences.length - 1) {
                decisionBoundaryForProtected = confidences[confidences.length - 1];
            } else {
                decisionBoundaryForProtected = confidences[numOfInstancesToMitigate];
            }
        }
    }

    private double[] adjustVotes(Instance inst, double[] votesForInstance) {
        if (isInstanceProtected(inst) && votesForInstance[positiveClass.getValue()] >= decisionBoundaryForProtected) {
            votesForInstance = new double[votesForInstance.length];
            votesForInstance[positiveClass.getValue()] = 1.0;
        }
        return votesForInstance;
    }

    private boolean isInstanceProtected(Instance inst) {
        return (int) inst.value(sensitiveAttribute.getValue()) == protectedValue.getValue();
    }

    private double getCumulativeStatisticalParity() {
        return (double) numOfNotProtectedPredictedPositive / (numOfNotProtected + L) - (double) numOfProtectedPredictedPositive / (numOfProtected + L);
    }

    private double getCumulativeEqualOpportunity() {
        return (double) numOfPositiveNotProtectedClassifiedPositive / (numOfPositiveNotProtected + L) - (double) numOfPositiveProtectedClassifiedPositive / (numOfPositiveProtected + L);
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == FairnessAwareAccuracyUpdatedEnsemble.class) {
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        } else {
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
        }
    }

    @Override
    public String getPurposeString() {
        return "Fairness Aware Accuracy Updated Ensemble";
    }

    private static class MisclassifiedExample{
        int timestamp;
        double confidence;

        MisclassifiedExample(int timestamp, double confidence){
            this.timestamp = timestamp;
            this.confidence = confidence;
        }
    }

}
