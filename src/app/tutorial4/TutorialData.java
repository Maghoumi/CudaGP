package app.tutorial4;

import com.sir_m2x.transscale.pointers.*;

import jcuda.Pointer;
import jcuda.driver.CUmodule;
import ec.gp.cuda.CudaData;

public class TutorialData extends CudaData {

	private static final long serialVersionUID = -5208644747900064178L;
	
	/** The size of the problem (i.e. the number of training instances) */
	protected int problemSize = -1;
	
	/** Training instances for x */
	protected double[] xValues = null;
	/** Training instances for y */
	protected double[] yValues = null;
	/** Expected output for each pair of (x, y) */
	protected double[] expectedOutput = null;
	
	/** Device instances for x */
	protected CudaDouble2D devX = null;
	/** Device instances for y */
	protected CudaDouble2D devY = null;
//	/** Device expected output for each pair of (x, y) */
//	protected CudaDouble2D devExpectedOutput = null;
	
	/** To be used by the X and Y terminals */
	public double currentX;
	public double currentY;
	public double currentExpectedResult;
	public double generatedResult;
	
	
	/**
	 * Initializes the arrays that hold the training instances
	 * There are two sets of arrays: one will be used by the CPU
	 * the other ones are suitable for being used by the GPU 
	 */
	public void initArrays(double[] xValues, double[] yValues, double[] expectedOutput) {
		this.xValues = xValues;
		this.yValues = yValues;
		this.expectedOutput = expectedOutput;
		
		/**
		 * Now initialize 2D arrays with the height of 1 (which would in fact be 1D arrays)
		 * These will hold the training instances on the GPU
		 * Lazy transfer will tell TransScale not to immediately allocated the arrays on the device
		 * This is necessary as we do not know at this point which GPU will work on these arrays
		 * (In a multi-gpu setup, this is a very important consideration)
		 */
		
		this.devX = new CudaDouble2D(problemSize, 1, 1, this.xValues, true);
		this.devY = new CudaDouble2D(problemSize, 1, 1, this.yValues, true);
//		this.devExpectedOutput = new CudaDouble2D(problemSize, 1, 1, this.expectedOutput, true);
	}

	/**
	 * Set the size of the problem (i.e. number of training instances)
	 * @param problemSize
	 */
	public void setProblemSize(int problemSize) {
		this.problemSize = problemSize;
	}
	
	/**
	 * @return	The size of the problem (i.e. the number of training instances)
	 */
	public int getProblemSize() {
		return this.problemSize;
	}
	
//	/**
//	 * @param index
//	 * @return	The value of X variable at the specified index in the training set
//	 */
//	public float getX(int index) {
//		return this.xValues[index];
//	}
//	
//	/**
//	 * @param index
//	 * @return	The value of Y variable at the specified index in the training set
//	 */
//	public float getY(int index) {
//		return this.yValues[index];
//	}
//	
	/**
	 * @param index
	 * @return	The expected result of the equation at the specified index
	 */
	public double getExpectedOutput(int index) {
		return this.expectedOutput[index];
	}

	@Override
	public Pointer[] getArgumentPointers() {
		Pointer[] result = new Pointer[2];
		
		result[0] = devX.toPointer();
		result[1] = devY.toPointer();
//		result[2] = devExpectedOutput.toPointer();
		
		return result;
	}

	@Override
	public void preInvocationTasks(CUmodule module) {
		devX.reallocate();
		devY.reallocate();
//		devExpectedOutput.reallocate();
	}

	@Override
	public void postInvocationTasks(CUmodule module) {
		devX.free();
		devY.free();
//		devExpectedOutput.free();
	}

	/**
	 * Based on the provided index, loads the correct training data
	 * to the public variables
	 * @param index	The index of the training instance to load
	 */
	public void loadCurrentTrainingData(int index) {
		this.currentX = xValues[index];
		this.currentY = yValues[index];
		this.currentExpectedResult = expectedOutput[index];
	}

	@Override
	public long[] getKernelInputPitchInElements() {
		return this.devX.getDevPitchInElements();
	}
}
