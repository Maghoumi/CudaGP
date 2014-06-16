package ec.gp.cuda;
/**
 * Explains a built-in type in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaType {
	
	/** The type of the most basic element of this type (int? float?..?) */
	protected String primitiveType;
	
	/** The size of the primitive type that constitute this data type */
	protected int primitiveSize;
	
	/** The name of the type */
	protected String typeName;
	
	/** The size of the type in bytes */
	protected int size;
	
	/**
	 * Creates a new type and assigns it the name of the type
	 * and the size of the type in bytes
	 * 
	 * @param primitiveType	The name of the most basic element of this type (int? float? ...?)
	 * @param typeName	The name of the type
	 * @param size	The size of the type in bytes
	 */
	public CudaType(String primitiveType, String typeName, int size) {
		this.primitiveType = primitiveType;
		this.typeName = typeName;
		this.size = size;		
	}
	
	/**
	 * Creates a new type and assigns it the name of the type that is supplied.
	 * This method will automatically try to determine the primitive type and the size
	 * of the type (based on a number in its name such as uchar4)
	 * Warning: If automatic detection is unsuccessful, this method will throw an exception
	 * Warning: automatic detection of size will only work if you adhere to CUDA's naming
	 * 			conventions: (e.g. uchar4, float4) => each type name ends with a number
	 * @param typeName
	 */
	public CudaType(String typeName) {
		this.typeName = typeName;		
		int multiplier = 1;	// filed multiplier
		int sizeInByte = 0;	// size of each field in bytes
		
		// Determine the primitive type
		if (typeName.contains("char")) {
			this.primitiveType = "char";
			sizeInByte = Byte.SIZE/8;
		}
		else if (typeName.contains("short")) {
			this.primitiveType = "short";
			sizeInByte = Short.SIZE/8;
		}
		else if (typeName.contains("int")) {
			this.primitiveType = "int";
			sizeInByte = Integer.SIZE/8;
		}
		else if (typeName.contains("long")) {
			this.primitiveType = "long";
			sizeInByte = Long.SIZE/8;
		}
		else if (typeName.contains("float")) {
			this.primitiveType = "float";
			sizeInByte = Float.SIZE/8;
		}
		else if (typeName.contains("double")) {
			this.primitiveType = "double";
			sizeInByte = Double.SIZE/8;
		}
		else
			throw new RuntimeException("The primitive type of type " + toString() + " was unknown");
		
		this.primitiveSize = sizeInByte;
		
		// Determine the multiplier (e.g. uchar16 => 16 x 1bytes = 16 bytes each field
		try {
			multiplier = Integer.parseInt("" + typeName.charAt(typeName.length() - 1));
		}
		catch (Exception e) {/*Do nothing! Stick to multiplier of 1*/}	
		
		this.size = multiplier * sizeInByte;
	}
	
	/**
	 * @return	The primitive type of this type (float? int?...?)
	 */
	public String getPrimitiveType() {
		return this.primitiveType;
	}
	
	/**
	 * @return	The size of the primitive type that constitute this data type
	 */
	public int getPrimitiveSize() {
		return this.primitiveSize;
	}
	
	/**
	 * @return	The name of the type
	 */
	public String getName() {
		return this.typeName;
	}
	
	/**
	 * @return	The size of the type in bytes
	 */
	public int getSize() {
		return this.size;
	}
	
	@Override
	public String toString() {
		return this.typeName;		
	}

}
