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
	public final static String P_INPUT = "input";
	public final static String P_OUTPUT = "output";
	public final static String P_CLASS = "class";
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
	
	/** The output of the kernel type and variable name */
	protected StringPair kernelOutput;

	@Override
	public void setup(EvolutionState state, Parameter base) {
		Parameter p = base.push(P_BASE);
		
		this.stackSize = state.parameters.getIntWithDefault(p.push(P_STACK_SIZE), null, this.stackSize);
		this.template = state.parameters.getStringWithDefault(p.push(P_TEMPLATE), null, this.template);
		this.name = state.parameters.getStringWithDefault(p.push(P_NAME), null, this.name);
		
		p = p.push(P_OUTPUT);
		String outputName = state.parameters.getString(p.push(P_NAME), null);
		if (outputName == null)
			state.output.fatal("Kernel output name is undefined");
		if (outputName.contains("*"))
			state.output.fatal("Kernel output type contains \"*\". Specify pointers using the type not the name! ");
		
		String outputType = state.parameters.getString(p.push(P_TYPE), null);
		if (outputType == null)
			state.output.fatal("Kernel output type is undefined");
		
		this.kernelOutput = new StringPair(outputName, outputType);
		
		p = p.pop();
		
		// Grab kernel argument information
		// First, how many arguments does the kernel have? (excluding expression, its length and fitness[])
		p = p.push(P_INPUT);
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
	 * @return The size of the stack
	 */
	public int getStackSize() {
		return stackSize;
	}

	/**
	 * @return The string containing the template for the CUDA kernel read from the file
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
	
	/**
	 * @return	The kernel output's portion of the kernel signature
	 */
	public String getKernelOutputSignature() {
		return String.format("%s %s,", this.kernelOutput.getValue(), this.kernelOutput.getKey());
	}
	
	/**
	 * @return	The name and the type of the kernel output variable
	 */
	public StringPair getKernelOutput() {
		return this.kernelOutput;
	}
	
	/**
	 * Creates an instance of KernelOutputData class using the parameters that the user
	 * has provided in ECJ. The new object will be initialized with the size of problemSize
	 * 
	 * @param state
	 * @param problemSize
	 * @return
	 */
	public KernelOutputData getOutputDataInstance(EvolutionState state, int problemSize) {
		Parameter p = new Parameter(CudaInterop.P_BASE).push(P_BASE).push(P_OUTPUT).push(P_CLASS);
		KernelOutputData instance = (KernelOutputData) state.parameters.getInstanceForParameter(p, null, KernelOutputData.class);
		instance.init(problemSize);	
		
		return instance;
	}

	
	/**
	 * Creates an array of KernelOutputData[] with the specified size.
	 * Each element in the created array is properly instantiated using
	 * the parameters that the user has supplied in the ECJ's parameter file
	 * 	
	 * @param state	ECJ's EvolutionState object
	 * @param problemSize	Size of the problem (i.e. the number of training instances)
	 * @param size	Size of the instantiated array 
	 * @return
	 */
	public KernelOutputData[] instantiateOutputArray(EvolutionState state, int problemSize, int size) {
		KernelOutputData[] result = new KernelOutputData[size];
		for (int i = 0 ; i < size ; i++) 
			result[i] = getOutputDataInstance(state, problemSize);
		
		return result;
	}

}
