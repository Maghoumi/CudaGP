package ec.gp.cuda;

import java.util.LinkedList;
import java.util.List;

import ec.EvolutionState;
import ec.Singleton;
import ec.util.Parameter;

/**
 * A helper class that stores the kernel information that is available
 * in ECJ's parameter file. 
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaKernelInfo implements Singleton {
	private static final long serialVersionUID = -2015666254866537775L;
	
	public final static String P_BASE = "kernel";
	public final static String P_ARGS = "args";
	public final static String P_NAME = "name";
	public final static String P_SIZE = "size";
	public final static String P_TYPE = "type";
	public final static String P_TEMPLATE = "template";
	public final static String P_STACK_SIZE = "stack-size";
	
	/** The stack size to use in the kernel */
	protected int stackSize = 128;
	
	/** The path to the CU file containing the evaluation kernel template */
	protected String template = "bin/ec/gp/cuda/kernel/evaluator-template.cu";
	
	/** The name of the CUDA kernel to call for evaluation purposes */
	protected String name = "evaluate";	
	
	/** List of kernel arguments along with their types */
	protected List<StringPair> argsList = new LinkedList<>();

	@Override
	public void setup(EvolutionState state, Parameter base) {
		Parameter p = base.push(P_BASE);
		
		this.stackSize = state.parameters.getIntWithDefault(p.push(P_STACK_SIZE), null, this.stackSize);
		this.template = state.parameters.getStringWithDefault(p.push(P_TEMPLATE), null, this.template);
		this.name = state.parameters.getStringWithDefault(p.push(P_NAME), null, this.name);
		
		// Grab kernel argument information
		// First, how many arguments does the kernel have? (excluding expression, its length and fitness[])
		p = p.push(P_ARGS);
		int numArgs = state.parameters.getIntWithDefault(p.push(P_SIZE), null, 0);
		
		if (numArgs < 0)
			state.output.fatal("Kernel cannot have negative number of arguments");
		
		// Parse the name/type pairs
		for (int i = 0 ; i < numArgs ; i++) {
			p = p.push("" + i);
			
			String argName = state.parameters.getString(p.push(P_NAME), null);
			String argType = state.parameters.getString(p.push(P_TYPE), null);
			
			if (argName == null) {
				state.output.fatal("Argument name for argument " + i + " is null");
			}
			
			if (argType == null) {
				state.output.fatal("Argument type for argument " + i + " is null");
			}
			
			if (argName.contains("*")) {
				state.output.fatal("Argument name for argument " + i + " contains \"*\". Specify pointers using the type not the name! ");
			}
			
			argType = argType.replace("-", " ");	// for const-int situations ==> const int
			
			// Add to the list of args
			argsList.add(new StringPair(argName, argType));
			
			p = p.pop();
		} // end-for
	}

	/**
	 * @return the stackSize
	 */
	public int getStackSize() {
		return stackSize;
	}

	/**
	 * @return the template
	 */
	public String getTemplate() {
		return template;
	}

	/**
	 * @return The name of the CUDA kernel to call for evaluation purposes
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * The list of the kernel arguments along with their types
	 * @return
	 */
	public List<StringPair> getArgsList() {
		return this.argsList;
	}
	
	/**
	 * @return The kernel arguments' portion of the kernel signature
	 */
	public String getKernelArgSignature() {
		StringBuilder result = new StringBuilder();
		
		for (StringPair pair : this.argsList) {
			result.append(String.format("%s %s, ", pair.getValue(), pair.getKey()));
		}
		
		return result.toString();
	}

}
