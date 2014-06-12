package ec.gp.cuda;

/**
 * Explains a built-in type in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaType {
	
	/** The name of the type */
	protected String typeName;
	
	/** The size of the type in bytes */
	protected int size;
	
	/**
	 * Creates a new type and assigns it the name of the type
	 * and the size of the type in bytes
	 * 
	 * @param typeName	The name of the type
	 * @param size	The size of the type
	 */
	public CudaType(String typeName, int size) {
		this.typeName = typeName;
		this.size = size;		
	}
	
	@Override
	public String toString() {
		return this.typeName;		
	}

}
