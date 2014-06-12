package app.tutorial4;

import ec.gp.cuda.KernelOutputData;

public class OutputData extends KernelOutputData {

	protected double[] output;
	
	@Override
	public void init(int count) {
		this.output = new double[count];
	}

	@Override
	public Object getUnderlyingData() {
		return this.output;
	}

	@Override
	public void setValueAt(int index, Object value) {
		this.output[index] = (Double)value;
	}

	@Override
	public Object getValueAt(int index) {
		return this.output[index];
	}

	@Override
	public void copyValues(Object sourceArray, int start, int length) {
		double[] kernelOutput = (double[]) sourceArray;
		
		System.arraycopy(kernelOutput, start, output, 0, length);
	}

}
