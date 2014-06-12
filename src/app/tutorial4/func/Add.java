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
import ec.gp.cuda.CudaTypes;
import ec.gp.cuda.InstructionSequencer;

public class Add extends CudaNode {
    public String toString() {
    	return "+";
    }

    public int expectedChildren() {
    	return 2;
    }

    public void eval(final EvolutionState state, final int thread, final GPData input,
    		final ADFStack stack, final GPIndividual individual, final Problem problem) {
    	double result;
        TutorialData data = ((TutorialData)(input));

        children[0].eval(state,thread,input,stack,individual,problem);
        result = data.generatedResult;

        children[1].eval(state,thread,input,stack,individual,problem);
        data.generatedResult = result + data.generatedResult;
    }

	@Override
	public InstructionSequencer getCudaAction() {
		InstructionSequencer result = new InstructionSequencer();
		result.pop(CudaTypes.DOUBLE, "second");
		result.pop(CudaTypes.DOUBLE, "first");
		result.verbatim("double final = first + second;");
		result.push("final");		
		return result;
	}
}

