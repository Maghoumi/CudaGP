package ec.gp.cuda;

/**
 * Defines the interface for a class that wants to provide methods to
 * manipulate and assign outputs of the CUDA kernel to a concrete data type.
 * You should subclass this class and the subclass must not have any constructors
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class KernelOutputData {
	
	public KernelOutputData() {
		// Constructor must always be EMPTY
		// Required for ECJ's reflection tools to work properly
	}
	
	public abstract void init(int count);
	
	public abstract Object getUnderlyingData();
	
	public abstract void setValueAt(int index, Object value);
	
	public abstract Object getValueAt(int index);
	
	/**
	 * Copies the values of the sourceArray to the underlying collection
	 * holding the data at the specified start index with the specified length 
	 * 
	 * @param sourceArray	The array to copy values from
	 * @param start		The start index in the sourceArray
	 * @param length	The length to copy
	 */
	public abstract void copyValues(Object sourceArray, int start, int length);
}
