/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package app.tutorial4.func;
import app.tutorial4.TutorialData;
import ec.*;
import ec.gp.*;
import ec.gp.cuda.CudaNode;
import ec.gp.cuda.InstructionSequencer;

public class X extends CudaNode {
    public String toString() {
    	return "x";
    }

    public int expectedChildren() {
    	return 0;
    }

    public void eval(final EvolutionState state, final int thread, final GPData input,
    		final ADFStack stack, final GPIndividual individual, final Problem problem) {
        TutorialData data = ((TutorialData)(input));
        data.generatedResult = data.currentX;
    }

	@Override
	public InstructionSequencer getCudaAction() {
		InstructionSequencer result = new InstructionSequencer();
		String x = result.getArgumentForCurrentThread("x");	// FIXME make "x" a constant somewhere
		result.push(x);
		
		return result;
	}
}