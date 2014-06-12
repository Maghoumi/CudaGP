package ec.gp.cuda;

/**
 * This class aims to provide easy access to the kernel side features.
 * Using instances of this class, one could define the sequence of instructions
 * that should be executed in order to evaluate a node in the GP tree using CUDA.
 * Each sequence must be a valid CUDA-C expression. These expressions are then added
 * to the kernel code and the kernel code is compiled using these expressions.
 * A GP tree can be converted to a postfix expression. This expression can be easily
 * evaluated using a switch-case based block in the CUDA kernel. Instruction sequencer
 * will help defining the micro-operations that are required for evaluating a node.
 * For example, when implementing a mathematical "Add" function, the following micro
 * instructions should be performed:
 * 
 * float b = pop();
 * float a = pop();
 * float result = a + b;
 * push (result);
 * 
 * The instruction sequencer will help build these operations for each CudaNode in
 * the GP language.
 * 
 * @author Mehran Maghoumi
 *
 */
public class InstructionSequencer {
	
	/** IMPORTANT: change the name here if the original name was changed in the template */
	protected static final String BLOCK_INDEX = "blockIndex";	// TODO put them in ECJ's param maybe??
	/** IMPORTANT: change the name here if the original name was changed in the template */
	protected static final String THREAD_INDEX = "threadIndex";	// TODO put them in ECJ's param maybe??
	/** IMPORTANT: change the name here if the original name was changed in the template */
	protected static final String MAPPED_THREAD_INDEX = "tid";	// TODO put them in ECJ's param maybe??
	/** IMPORTANT: change the name here if the original name was changed in the template */
	protected static final String EVAL_RESULT_VARIABLE = "eval_result";
	
	/** Line separator for Java < 7 */
	protected static String lineSeparator = System.getProperty("line.separator");
	
	/** Holds the synthesized sequence */
	protected StringBuilder sequence = new StringBuilder();
	
	/**
	 * Generates the C sequence that would pop a value from the stack
	 * and stores it in the provided variable name; i.e.
	 * 
	 * pop(CudaTypes.FLOAT, "a") is equivalent to "float a = pop();" 
	 * 
	 * @param	type	The type of the variable to use for storing the popped value
	 * @param varName	The name of the variable to store the popped value into
	 */
	public void pop(CudaType type, String varName) {
		sequence.append(String.format("%s %s;pop(%s);%s", type, varName, varName, lineSeparator));
	}
	
	/**
	 * Generates the C sequence that would push the specified variable on the
	 * stack.
	 * 
	 * @param varName	The name of the variable the value of which is to be pushed
	 * 					into the stack
	 */
	public void push(String varName) {
		sequence.append(String.format("push(%s);%s",varName, lineSeparator));
	}
	
	/**
	 * Appends a verbatim code to the sequence of the code that is already synthesized.
	 * 
	 * @param code	The code that should be appended to the synthesized string.
	 */
	public void verbatim(String code) {
		sequence.append(code + lineSeparator);
	}
	
	/**
	 * @return	Returns the block index variable name in the kernel code
	 * 			This value is not the mapped value and represents the raw
	 * 			index of the block (i.e. blockDim.x)
	 */
	public String getBlockIndex() {
		return BLOCK_INDEX;
	}
	
	/**
	 * @return	Returns the thread index variable name in the kernel code
	 * 			This value is not the mapped value and represents the raw
	 * 			index of the thread (i.e. threadIdx.x)
	 */
	public String getThreadIndex() {
		return THREAD_INDEX;
	}
	
	/**
	 * @return	Returns the thread-to-data-index-mapped variable name in the kernel code
	 */
	public String getMappedThreadIndex() {
		return MAPPED_THREAD_INDEX;
	}
	
	/**
	 * @return	Returns the variable that holds the popped value from the stack at the end of the
	 * 			interpretation.
	 */
	public String getEvaluationResultVariable() {
		return EVAL_RESULT_VARIABLE;
	}
	
	/**
	 * Finds the kernel argument with the specified name and returns it
	 * If the argument is not found, null will be returned.
	 * 
	 * @param argName	The name of the argument to find
	 * @return
	 */
	public String getArgument(String argName) {
		CudaInterop interop = CudaInterop.getInstance();
		String type = interop.getArgumentType(argName);
		
		return type;
	}
	
	/**
	 * Returns the specified kernel argument for the specified index.
	 * NOTE: This method will not check if the specified argument is a pointer or not!
	 * 		 So be careful!
	 *  
	 * @param argName	The requested argument's name
	 * @return	A string containing the properly referenced argument to be used for the
	 * 			specified index or null if the argument does not exist
	 */
	public String getArgumentForIndex(String argName, String index) {
		String argument = getArgument(argName);
		
		if (argument == null) {
			return null;
		}
		
		return String.format("%s[%s]", argName, index);
	}
	
	/**
	 * Returns the specified kernel argument for the current thread (referenced with "tid").
	 * NOTE: This method will not check if the specified argument is a pointer or not!
	 * 		 So be careful!
	 *  
	 * @param argName	The requested argument's name
	 * @return	A string containing the properly referenced argument to be used for the
	 * 			current thread or null if the argument does not exist
	 */
	public String getArgumentForCurrentThread(String argName) {
		return getArgumentForIndex(argName, MAPPED_THREAD_INDEX);
	}
		
	/**
	 * @return	Returns the generated sequence in a string representation
	 */
	@Override
	public String toString() {
		return this.sequence.toString();
	}
	
}
