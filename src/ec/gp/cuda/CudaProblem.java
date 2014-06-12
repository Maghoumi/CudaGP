package ec.gp.cuda;

import ec.EvolutionState;
import ec.gp.GPProblem;
import ec.simple.SimpleProblemForm;
import ec.util.Parameter;

/**
 * An extension of ECJ's GPProblem which mandates having a problem size
 * parameter in the ECJ's parameter file. Users of this class can access
 * the number of training instances in the problem.
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class CudaProblem extends GPProblem implements SimpleProblemForm {
	private static final long serialVersionUID = -4206807496360216888L;
	
	public static final String P_SIZE = "size";	

	/** The size of the problem (i.e. the number of training instances) */
	protected int problemSize = -1;
	
	@Override
	public void setup(EvolutionState state, Parameter base) {
		super.setup(state, base);
		
		this.problemSize = state.parameters.getInt(base.push(P_SIZE), null);
	}
	
}
