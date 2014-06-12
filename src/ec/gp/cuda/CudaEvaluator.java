package ec.gp.cuda;

import java.util.ArrayList;
import java.util.List;

import app.tutorial4.MultiValuedRegression;
import ec.EvolutionState;
import ec.Subpopulation;
import ec.gp.GPIndividual;
import ec.gp.GPProblem;
import ec.simple.SimpleEvaluator;
import ec.simple.SimpleFitness;
import ec.simple.SimpleProblemForm;
import ec.util.Parameter;
import gnu.trove.list.array.TByteArrayList;

/**
 * An evaluator which works exactly like SimpleEvaluator but performs the
 * evaluations in CUDA. You should subclass this class, implement the setFitness()
 * method and provide the subclass's name to use as the evaluator in ECJ's parameter file.
 * 
 * @author Mehran Maghoumi
 * 
 */
public abstract class CudaEvaluator extends SimpleEvaluator {
	
	private static final long serialVersionUID = 2872338520868386315L;
	
	public static final String P_EVALUATION_METHOD = "evaluation-method";
	
	/** Keeps the expression list for each thread */
	protected List<List<TByteArrayList>> threadExpList;
	private boolean useCuda = false;
	
	// checks to make sure that the Problem implements SimpleProblemForm
	@Override
	public void setup(final EvolutionState state, final Parameter base) {
		super.setup(state, base);
		if (!(p_problem instanceof SimpleProblemForm))
			state.output.fatal("" + this.getClass()
					+ " used, but the Problem is not of SimpleProblemForm",
					base.push(P_PROBLEM));
		
		// Initialize the list of list of expressions
		
		threadExpList = new ArrayList<List<TByteArrayList>>(state.evalthreads);
		
		useCuda = state.parameters.getStringWithDefault(base.push(P_EVALUATION_METHOD), null, "cpu").toLowerCase().equals("gpu"); 
		
		
		String cudaMessage
						= "#########################\n"
				+ 		  "#   Using NVIDIA CUDA   #\n"
				+ 		  "#########################";
		
		String cpuMessage
						= "#################\n"
				+ 		  "#   Using CPU   #\n"
				+ 		  "#################";
		
		state.output.message(useCuda ? cudaMessage : cpuMessage);
		
		for(int i = 0 ; i < state.evalthreads ; i++) {
			threadExpList.add(new ArrayList<TByteArrayList>());
		}
		
	}

	/**
	 * A simple evaluator that doesn't do any coevolutionary evaluation.
	 * Basically it applies evaluation pipelines, one per thread, to various
	 * subchunks of a new population. Each thread is responsible for converting
	 * its own subchunk to a postfix expression.
	 */
	@Override
	public void evaluatePopulation(final EvolutionState state) {
		
		// Not using CUDA? What a shame... :-<
		if (!useCuda) {
			super.evaluatePopulation(state);
			return;
		}	

		int[] from = new int[state.evalthreads]; // starting index of this thread
		int[] to = new int[state.evalthreads];	// ending index of this thread
		
		int offset = 0;
		
		// These stuff should be done per subpopulation.
		for (Subpopulation sp : state.population.subpops) {
			CudaSubpopulation subPop = (CudaSubpopulation) sp;
			
			// Determine the working scope of each thread
			for (int i = 0 ; i < state.evalthreads ; i++) {
				List<GPIndividual> listOfInd = subPop.needEval.get(i);
				
				from[i] = offset;
				to[i] = from[i] + listOfInd.size() - 1;
				offset += listOfInd.size();
			}
			
			if (state.evalthreads == 1)
				traversePopChunk(state, subPop, 0);
			else {
				Thread[] t = new Thread[state.evalthreads];
	
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					ByteTraverseThread r = new ByteTraverseThread();
					r.threadnum = y;
					r.me = this;
					r.state = state;
					r.subPop = subPop;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
	
			}
			
			// Call the CUDA kernel using the defined problem data
			CudaData data = (CudaData) ((GPProblem)p_problem).input;
			KernelOutputData[] outputs = CudaInterop.getInstance().evaluatePopulation(state, threadExpList, data);
			
			// call the assignFitness and assign fitnesses to each individual
			if (state.evalthreads == 1) {
				threadExpList.get(0).clear();
				assignFitness(state, subPop, 0, outputs, from, to);
			}
			else {
				Thread[] t = new Thread[state.evalthreads];
				
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					// first clean this threads expressions
					
					threadExpList.get(y).clear();
					
					OutputAssignmentThread r = new OutputAssignmentThread();
					r.threadnum = y;
					r.me = this;
					r.state = state;
					r.subPop = subPop;
					r.outputs = outputs;
					r.from = from;
					r.to = to;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
			}
		} // end-for (subpopulation)
		//Finished! :-)
	}

	protected void traversePopChunk(EvolutionState state, CudaSubpopulation subPop, int threadnum) {
		// Get the unevaluateds for the current thread
		List<GPIndividual> myNeedEvals = subPop.needEval.get(threadnum);
		List<TByteArrayList> myExpList = threadExpList.get(threadnum);
		
		// Walk through my individuals and convert them to byte[] expressions
		// and store them in myExpList
		for(GPIndividual ind : myNeedEvals) {
			CudaNode root = (CudaNode) ind.trees[0].child;
			byte[] exp = root.makePostfixExpression();
			// Add this expression to the list
			myExpList.add(new TByteArrayList(exp));
		}
	}
	
	/**
	 * Passes the calculated CUDA outputs to each individual in the population for fitness evaluation
	 * 
	 * @param state
	 * @param threadnum
	 * @param startIndex
	 * @param endIndex
	 */
	protected void assignFitness(EvolutionState state, CudaSubpopulation subPop, int threadnum, KernelOutputData[] fitnesses, int[] from, int[] to) {
		List<GPIndividual> myUnevals = subPop.needEval.get(threadnum); // get my unevaluated individuals
		CudaProblem problem = (CudaProblem) p_problem;
		
		int indIndex = 0;	// hold the index to my individuals
		for (int i = from[threadnum] ; i <= to[threadnum] ; i++) {
			GPIndividual currentInd = myUnevals.get(indIndex++);
			
			// Ask the problem to assign a fitness value to this individual based
			// on the outputs of the kernel
			problem.assignFitness(state, currentInd, fitnesses[i]);
			
			/**
			double cpuResult = ((MultiValuedRegression)p_problem).getFitness(state, currentInd, threadnum);			
			final float EPSILON = (float) 0.01; 			
			if (Math.abs(cpuResult - fitnesses[i]) > EPSILON) {
				System.out.println(String.format("Different! Expected %f but got %f", cpuResult, fitnesses[i]));
			}*/
			
			currentInd.evaluated = true; // Current individual is now evaluated :-)
		}
		
		// Clean my unevaluateds
		myUnevals.clear();
	}
	
	/**
	 * Sets the fitness of the given individual with respect to its fitness type
	 * to the provided fitness value.
	 * You may have used a specific fitness class as the fitness type of your individual.
	 * This poses some difficulty when assigning a fitness to an individual because based on the
	 * class of the fitness (KozaFitness, SimpleFitness, etc.) there are different methods for
	 * assigning a fitness value to an individual.
	 * You should implement this method based on the type of fitness you specified in the
	 * ECJ's parameter file
	 * 
	 * @param individual	The individual to assign a fitness to
	 * @param fitness	The calculated value of the fitness of the individual.
	 */
	public abstract void setFitness(EvolutionState state, GPIndividual individual, double fitness);

	/** A private helper class for implementing multithreaded byte traversal */
	private class ByteTraverseThread implements Runnable {
		public CudaEvaluator me;
		public EvolutionState state;
		public int threadnum;
		public CudaSubpopulation subPop;
		
		public synchronized void run() {
			me.traversePopChunk(state, subPop, threadnum);
		}
	}
	
	/** A private helper class for implementing multithreaded fitness assignment */
	private class OutputAssignmentThread implements Runnable {
		public CudaEvaluator me;
		public EvolutionState state;
		public int threadnum;
		public int[] from;
		public int[] to;
		public KernelOutputData[] outputs;
		public CudaSubpopulation subPop;
		
		public synchronized void run() {
			me.assignFitness(state, subPop, threadnum, outputs, from, to);
		}
	}
}
