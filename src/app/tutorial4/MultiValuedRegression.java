package app.tutorial4;
import com.sir_m2x.transscale.pointers.CudaPrimitive2D;

import ec.util.*;
import ec.*;
import ec.gp.*;
import ec.gp.cuda.CudaProblem;
import ec.gp.koza.*;

public class MultiValuedRegression extends CudaProblem {

	private static final long serialVersionUID = 1755954437542996060L;
	
	/** It's here so as to prevent excessive casting */
	protected TutorialData problemData = null;
	
    public void setup(final EvolutionState state, final Parameter base) {
    	super.setup(state, base);
    	
    	// Prepare training instances
    	this.problemData = (TutorialData) this.input;
    	// Obtain the problem size
    	this.problemData.setProblemSize(problemSize);
    	
    	double[] x = new double[problemSize];
    	double[] y = new double[problemSize];
    	double[] expectedOutput = new double[problemSize];
    	
    	// Generate some training instances according to the desired formula
    	for (int i = 0 ; i < problemSize ; i++) {
    		double currentX = state.random[0].nextDouble();
    		double currentY = state.random[0].nextDouble();
    		double currentOutput = currentX*currentX*currentY + currentX*currentY + currentY;
    		
    		x[i] = currentX;
    		y[i] = currentY;
    		expectedOutput[i] = currentOutput;    		
    	}
    	
    	// Now initialize the problem data arrays
    	this.problemData.initArrays(x, y, expectedOutput);
    }
        
    public void evaluate(final EvolutionState state, final Individual ind, final int subpopulation, final int threadnum) {
    	if (ind.evaluated)	// don't bother reevaluating
    		return;
    	
    	double sum = 0;
    	double result;
    	int hits = 0;
    	final double PROBABLY_ZERO = 1.11E-15;
        
        // Evaluate the tree using all training instances
        for (int i= 0 ; i<problemData.getProblemSize() ; i++) {
        	problemData.loadCurrentTrainingData(i);
            ((GPIndividual)ind).trees[0].child.eval(state,threadnum,problemData,stack,((GPIndividual)ind),this);
            
            result = Math.abs(problemData.currentExpectedResult - problemData.generatedResult);
            
            if (result < PROBABLY_ZERO)
            	result = 0.0;
            
            if (result <= 0.01)
            	hits++;
            
            sum += result;
        }

        // the fitness better be KozaFitness!
        KozaFitness f = ((KozaFitness)ind.fitness);
        f.setStandardizedFitness(state, sum);
        f.hits = hits;
        ind.evaluated = true;
    }
    
    public double getFitness(final EvolutionState state, final Individual ind, final int threadnum) {
    	double sum = 0;
    	double result;
        
        // Evaluate the tree using all training instances
        for (int i= 0 ; i<problemData.getProblemSize() ; i++) {
        	problemData.loadCurrentTrainingData(i);
            ((GPIndividual)ind).trees[0].child.eval(state,threadnum,problemData,stack,((GPIndividual)ind),this);
            
            result = Math.abs(problemData.currentExpectedResult - problemData.generatedResult);
            sum += result;
        }
        
        return sum;
    }
    
    public static void main(String[] args) {
    	String[] arguments = new String[] {"-file", "bin/app/tutorial4/tutorial4.params"};
    	CudaPrimitive2D.usePitchedMemory(true);	// Do use the pitched memory
    	ec.Evolve.main(arguments);
    }
}

