package ec.gp.cuda;

import ec.EvolutionState;
import ec.Singleton;
import ec.gp.cuda.kernel.KernelArgument;
import ec.util.Parameter;
import gnu.trove.list.array.TByteArrayList;

import java.io.*;
import java.util.*;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

import com.sir_m2x.transscale.*;
import com.sir_m2x.transscale.pointers.*;

/**
 * Provides necessary tools required for either making CUDA kernel calls
 * or preparing data that are needed for calling the kernels or enable
 * ECJ to run on CUDA.
 * 
 * This class uses the singleton design pattern (not to be confused with
 * ECJ's singleton concept although this class also implements the singleton 
 * interface)
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaInterop implements Singleton {
	
	private static final long serialVersionUID = -1231586517576285043L;
	
	public final static String P_BASE = "cuda";
    public final static String P_PROBLEM_SIZE = "eval.problem.size"; //FIXME
    public final static String P_RECOMPILE = "recompile";
    public final static String P_GEN_DEBUG = "gen-debug";
    public final static String P_EVAL_BLOCK_SIZE = "eval-block-size";
    
    /*
     * The tags that are replaced in the kernel template are all
     * defined in here. If you change a tag here, it must also be
     * changed in the evaluator template and vice versa.
     */
    private static final String T_STACK_SIZE = "/*@@stack-size@@*/";
    private static final String T_PROBLEM_SIZE = "/*@@problem-size@@*/";
    private static final String T_BLOCK_SIZE = "/*@@block-size@@*/";
    private static final String T_KERNEL_ARGS = "/*@@kernel-args@@*/";
    private static final String T_INTERPRETER = "/*@@interpreter@@*/";
    private static final String T_KERNEL_OUT = "/*@@kernel-out@@*/";
    private static final String T_KERNEL_OUT_NAME = "/*@@kernel-out-name@@*/";
    private static final String T_KERNEL_OUT_TYPE = "/*@@kernel-out-type@@*/";
    private static final String T_KERNEL_OUT_TYPE_NOPOINTER = "/*@@kernel-out-type-nopointer@@*/";
	
	/** Holds the single instance of this class during program execution */
	protected static CudaInterop instance = null;
	
	/** The problem size to use in the kernel */
	protected int problemSize = -1;
	
	/** Recompile flag, if true, the kernel file will be compiled no matter what */
	protected boolean recompile;
	
	/** Debugging flag, if true, the kernel file will be created with debugging information */
	protected boolean genDebug;
	
	/** Number of threads in a CUDA thread block */
	protected int evaluateBlockSize = 512;
	
	/** The code fragment for the expression parser in the CUDA kernel */
	protected String interpreterCode;
	
	/** The output file name of the CU file */
	protected String outputFilename = "bin/ec/gp/cuda/kernel/evaluator.cu";
	
	/** Holds an instance to a KernelInfo object which helps in determining various properties of the kernel */
	protected CudaKernelInfo kernelInfo = null;
	
	/**
	 * @return	Returns the single instance that exists from this class
	 * 			If no instance exists, an instance will be created
	 */
	public synchronized static CudaInterop getInstance() {
		if (instance == null) {
			instance = new CudaInterop();
		}
		
		return instance;
	}
	
	protected CudaInterop() {
		// Dummy constructor 
	}
	
	/**
	 * Must parse the parameter file and extract useful stuff
	 * @param base	Garbage base parameter (base will be created because it's easier)
	 */
	@Override
	public void setup(EvolutionState state, Parameter base) {
		Parameter p = new Parameter(P_BASE);
		
		this.problemSize = state.parameters.getInt(new Parameter(P_PROBLEM_SIZE), null);
		this.recompile = state.parameters.getBoolean(p.push(P_RECOMPILE), null, true);
		this.genDebug = state.parameters.getBoolean(p.push(P_GEN_DEBUG), null, false);
		this.evaluateBlockSize = state.parameters.getIntWithDefault(p.push(P_EVAL_BLOCK_SIZE), null, 512);
		
		// Setup the KernelInfo object using the parameter file
		this.kernelInfo = new CudaKernelInfo();
		this.kernelInfo.setup(state, p);
	}
	
	/**
	 * Prepares and compiles the kernel code based on the parameters and the language
	 * set defined in ECJ's parameter file 
	 */
	public void prepareKernel(EvolutionState state) {
		String template = null;
		
		try {
			template = readFile(kernelInfo.getTemplate());
		}
		catch (IOException e) {
			state.output.fatal("An error occured while preparing the template kernel: " + e.getMessage());
		}
		
		// Replace the placeholders in the template with concrete values
		template = template.replace(T_STACK_SIZE, "" + kernelInfo.getStackSize());
		template = template.replace(T_PROBLEM_SIZE, "" + this.problemSize);
		template = template.replace(T_BLOCK_SIZE, "" + this.evaluateBlockSize);
		template = template.replace(T_INTERPRETER, this.interpreterCode);
		//TODO support arch=? and sm=? in parameter file (as well as other CUDA parameters)
		
		template = template.replace(T_KERNEL_ARGS, kernelInfo.getKernelArgSignature());
		template = template.replace(T_KERNEL_OUT, kernelInfo.getKernelOutputSignature());
		KernelArgument kernelOutput = kernelInfo.getKernelOutput();
		String name = kernelOutput.getName();
		String type = kernelOutput.getType().getName();
		template = template.replace(T_KERNEL_OUT_NAME, name);
		template = template.replace(T_KERNEL_OUT_TYPE, type);
		template = template.replace(T_KERNEL_OUT_TYPE_NOPOINTER, type.replace("*", ""));
		
		
		// Write the temporary file
		try {
			writeFile(outputFilename, template);
		} catch (IOException e) {
			state.output.fatal("An error occured while writing evaluator CU file: " + e.getMessage());
		}
		
		String ptxFile = null;
		
		try {
			ptxFile = TransScale.preparePtxFile(outputFilename, recompile, genDebug);
		} catch (IOException e) {
			state.output.fatal("An error occured while compiling the kernel: " + e.getMessage());
		}
		
		// Load kernel in TransScale
		Kernel kernel = new Kernel();
		kernel.ptxFile = new File(ptxFile);
		Map<String, String> functionMapping = new HashMap<>();
		functionMapping.put(this.kernelInfo.getName(), this.kernelInfo.getName());
		kernel.functionMapping = functionMapping;
		TransScale.getInstance().addKernel(kernel);
	}

	/**
	 * Sets the interpreter part of the code of the CUDA kernel. The interpreter
	 * is responsible for parsing the expression tree. This code fragment
	 * is made based on the language definition of ECJ's parameter file and will
	 * be passed to this function by CudaFunctionSet
	 * 
	 * @param interpreterCode	The generated code fragment for the interpreter
	 */
	public synchronized void setInterpreterCode(String interpreterCode) {
		this.interpreterCode = interpreterCode;
	}

	/**
	 * Evaluates the list of the provided individuals and returns their fitness as a float
	 * array
	 * 
	 * @param expressions	The expression list of each ECJ thread
	 * @param data	A subclass of CudaData that contains the data that are required for
	 * 				calling the evaluation kernel (usually the inputs for the GP trees or
	 * 				training instances)
	 * @return	A KernelOutputData array that contains the output value of each individual on a training instance
	 */
	public KernelOutputData[] evaluatePopulation(EvolutionState state, List<List<TByteArrayList>> expressions, final CudaData data) {
		// First determine how many unevals we have in total
		int indCount = 0;
		int maxExpLengthtmp = 0;

		for (List<TByteArrayList> thExps : expressions) {
			indCount += thExps.size();

			// Determine the longest expression
			for (TByteArrayList exp : thExps)
				if (exp.size() > maxExpLengthtmp)
					maxExpLengthtmp = exp.size();
		}
		
		final int maxExpLength = maxExpLengthtmp;
		
		// Now that I know how many individuals I have, each one should get a  

		// Convert expressions to byte[]
		byte[] population = new byte[indCount * maxExpLength];
		int i = 0;

		for (List<TByteArrayList> thExps : expressions) {
			for (TByteArrayList currExp : thExps) {
				int length = currExp.size();
				currExp.toArray(population, 0, i * maxExpLength, length);
				i++;
			}
		}
		
		// Break population into chunks, each chunk is evaluated by a single GPU
		TransScale scaler = TransScale.getInstance();
		int gpuCount = scaler.getNumberOfDevices();
		KernelOutputData[] results = kernelInfo.instantiateOutputArray(state, problemSize, indCount);
		
		// Create gpuCount number of threads. Each thread will call "waitFor" for a single job 
		Thread[] gpuInteropThreads = new Thread[gpuCount];
		
		int arrayCpyOffset = 0;	// Offset variable
//		int assignmentOffset = 0;	// Offset variable
		
		List<byte[]> chunks = new ArrayList<>();
		
		for (i = 0 ; i < gpuCount ; i++) {
			// Calculate the output share and population share of the thread
			// that will be evaluating on the i'th GPU
			final int thisOutputShare;
			int thisPopShare;
			
			// *Evenly* divide the number of individuals, though the last one gets a larger portion
			if (i == gpuCount - 1) {
				thisOutputShare = indCount - i * (indCount / gpuCount);
			}
			else {
				thisOutputShare = indCount / gpuCount;
			}
			
			final GpuRunnerThread runner = new GpuRunnerThread();
			runner.scaler = scaler; 
			runner.destination = results;			
//			runner.start = assignmentOffset;
			runner.myShare = thisOutputShare;	// Set the length of the copy-back operation
			runner.problemSize = this.problemSize;
			
			thisPopShare = thisOutputShare * maxExpLength;
			
			// Now allocate a 2D array of height [my portion of indCount] and width [problemSize]
			// Each row represents an individual while each column represents the
			// output value of the said individual for the corresponding training instance
			String outputType = kernelInfo.getKernelOutput().getType().getPrimitiveType();	// get rid of the pointer so as to prevent confusion for CudaType constructor		
			CudaType outputCudaType = new CudaType(outputType);
			final CudaPrimitive2D chunkOutput = instantiateForType(outputCudaType, problemSize, thisOutputShare);	// Allocate the output pointer for this portion 	
			
			byte[] popChunk  = new byte[thisPopShare];
			System.arraycopy(population, arrayCpyOffset, popChunk, 0, thisPopShare);	// Copy this GPU's chunk of expressions
			chunks.add(popChunk);
			final CudaByte2D devExpression = new CudaByte2D(thisPopShare, 1, 1, popChunk, true);	// Allocate device expression pointer
			
			arrayCpyOffset += thisPopShare;
//			assignmentOffset += thisOutputShare;
			
			Trigger pre = new Trigger() {
				
				@Override
				public void doTask(CUmodule module) {
					// Do the user's pre-invocation tasks
					data.preInvocationTasks(module);
					
					chunkOutput.reallocate();
					devExpression.reallocate();
				}
			};
			
			Trigger post = new Trigger() {
				
				@Override
				public void doTask(CUmodule module) {
					devExpression.free();
					chunkOutput.refresh();					
					runner.kernelResult = chunkOutput.getUnclonedArray();	// Assign the kernel output array to the raw result holder of this thread
					chunkOutput.free();
					
					// Do the user's post-invocation tasks
					data.postInvocationTasks(module);
				}
			}; 
			
			// Create a kernel job
			KernelInvoke kernelJob = new KernelInvoke();
			
			kernelJob.functionId = kernelInfo.getName();
			kernelJob.preTrigger = pre;
			kernelJob.postTrigger = post;
			
			kernelJob.gridDimX = thisOutputShare;
			kernelJob.gridDimY = 1;
			
			kernelJob.blockDimX = this.evaluateBlockSize;
			kernelJob.blockDimY = 1;
			kernelJob.blockDimZ = 1;
			
			kernelJob.argSetter = new KernelArgSetter() {
				
				@Override
				public Pointer getArgs() {
					
					// Prepare the pointer to pointer to arguments
					Pointer[] extraArgumentPointers = data.getArgumentPointers();
					int userArgsCount = extraArgumentPointers.length;
					
					/*
					 * We need 5 additional spots for the following arguments defined in the
					 * kernel template file:
					 * 		inputPitch
					 * 		outputPitch
					 * 		individuals
					 * 		indCounts
					 * 		maxLength
					 * 
					 */
					Pointer[] kernelArguments = new Pointer[userArgsCount + 6];
					
					// Copy user's pointers
					System.arraycopy(extraArgumentPointers, 0, kernelArguments, 0, userArgsCount);
					kernelArguments[userArgsCount++] = Pointer.to(data.getKernelInputPitchInElements());	// Input pitch
					kernelArguments[userArgsCount++] = chunkOutput.toPointer();	// Output argument
					kernelArguments[userArgsCount++] = Pointer.to(chunkOutput.getDevPitchInElements());	// Output pitch
					kernelArguments[userArgsCount++] = devExpression.toPointer();	// This GPU's portion of the expression
					kernelArguments[userArgsCount++] = Pointer.to(new int[] {thisOutputShare});	// This GPU's portion of the output
					kernelArguments[userArgsCount] = Pointer.to(new int[] {maxExpLength});	// Maximum length of each expression
					
					return Pointer.to(kernelArguments);
				}
			};
					
			
			kernelJob.id = "Popsize: " + indCount + " (array length of  " + population.length + ") my share is " + thisOutputShare + " processing " + thisPopShare + " maxExpLength:" + maxExpLength;
			
			runner.kernelJob = kernelJob;
			gpuInteropThreads[i] = new Thread(runner);
			// Run this job on this thread, wait for the job to finish, then copy back the fitness
			gpuInteropThreads[i].start();
		}
		
		// Wait for auxiliary threads to finish their job
		for (i = 0 ; i < gpuCount ; i++) {
			try {
				gpuInteropThreads[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}		
		
		// No need to merge anything! The threads have done that :-)
		return results;
	}
	
	/**
	 * Returns the type of the specified argument of the CUDA kernel
	 * @param argName	The name of the argument to look for
	 * @return	A string representing the type of the argument
	 */
	public String getArgumentType(String argName) {
		List<KernelArgument> argList = kernelInfo.getArgsList();
		
		for (KernelArgument pair : argList) {
			if (pair.getName().equals(argName)) {
				return pair.getName();
			}
		}
		
		return null;		
	}	
	
	/**
	 * Reads the file in the specified path and converts it to a string
	 * @param path	The path to the file to read
	 * @return	A string containing the file content
	 * @throws IOExcepton 
	 */
	private String readFile(String path) throws IOException {
		File f = new File(path);
		StringBuilder result = new StringBuilder();
		String lineSeparator = System.getProperty("line.separator");	// System's line separator
		
		BufferedReader br = new BufferedReader(new FileReader(f));
		String line = "";
		
		while ((line = br.readLine()) != null) {
			result.append(line + lineSeparator);
		}
		
		// Remove the extra new line at the very end
		result.replace(result.length() - lineSeparator.length(), result.length(), "");
		
		br.close();
		
		return result.toString();
	}
	
	/**
	 * Writes the specified string to the specified file
	 * @param path
	 * @param content
	 * @throws IOException
	 */
	private void writeFile(String path, String content) throws IOException {
		PrintWriter writer = new PrintWriter(path);
		writer.print(content);
		writer.close();
	}
	
	/**
	 * Utility method that would instantiate a proper CudaPrimitive2D type using the information
	 * in the CudaType value supplied.
	 * 
	 * @param type	The CudaType value to use for determining the correct type to instantiate
	 * @param width	The width of the 2D memory
	 * @param height	The height of the 2D memory
	 * @return	A correctly allocated CudaPrimitive2D instance
	 */
	private static CudaPrimitive2D instantiateForType(CudaType type, int width, int height) {
		int numFields = type.getSize() / type.getPrimitiveSize();
		String primitiveType = type.getPrimitiveType();
		
		CudaPrimitive2D result = null;
		
		if (primitiveType.equals("char"))
			result = new CudaByte2D(width, height, numFields, new byte[width * height * numFields], true);	//FIXME in transscale, fix this crap!
		else if (primitiveType.equals("short"))
			result = new CudaShort2D(width, height, numFields, new short[width * height * numFields], true);
		else if (primitiveType.equals("int"))
			result = new CudaInteger2D(width, height, numFields, new int[width * height * numFields], true);
		else if (primitiveType.equals("long"))
			result = new CudaLong2D(width, height, numFields, new long[width * height * numFields], true);
		else if (primitiveType.equals("float"))
			result = new CudaFloat2D(width, height, numFields, new float[width * height * numFields], true);
		else if (primitiveType.equals("double"))
			result = new CudaDouble2D(width, height, numFields, new double[width * height * numFields], true);
		
		return result;
	}
	
	/**
	 * Helper thread that queues a job on GPU, waits for the job to finish and
	 * then obtains the results and copies its portion of the output to the final
	 * output.
	 * 
	 * @author Mehran Maghoumi
	 *
	 */
	private class GpuRunnerThread implements Runnable {
		
		public TransScale scaler;	// The TransScale instance to use
		public KernelInvoke kernelJob;	// The job to run on TransScale
		
		public KernelOutputData[] destination;	// Final place to store the output results of the kernel
		public Object kernelResult;	// The raw output results of the kernel that will be transferred to host
//		public int start;
		public int myShare;		// How many individuals am I handling?
		public int problemSize;	// Number of training instances (and by extension output elements generated by the kernel)

		@Override
		public void run() {
			scaler.queueJob(kernelJob);
			kernelJob.waitFor();
			
			// Copy my portion of the kernel output values :-)
			for (int i = 0 ; i < destination.length ; i++) {
				KernelOutputData item = destination[i];
				item.copyValues(kernelResult, i * problemSize, problemSize);
			}
		}
		
	}

}
