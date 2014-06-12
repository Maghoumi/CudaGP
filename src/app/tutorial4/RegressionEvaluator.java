package app.tutorial4;

import ec.EvolutionState;
import ec.gp.GPIndividual;
import ec.gp.cuda.CudaEvaluator;
import ec.gp.koza.KozaFitness;

public class RegressionEvaluator extends CudaEvaluator {

	@Override
	public void setFitness(EvolutionState state, GPIndividual individual, double fitness) {
		KozaFitness f = ((KozaFitness)individual.fitness);
        f.setStandardizedFitness(state, fitness);
	}

}
