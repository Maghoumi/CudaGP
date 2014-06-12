package ec.gp.cuda;

/**
 * Constants for sizes of built-in CUDA types including the
 * primitive types and vector types
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaTypes {
	// Primitive sizes
	public static final CudaType CHAR = new CudaType("char", "char", Byte.SIZE/8);
	public static final CudaType SHORT = new CudaType("short", "short", Short.SIZE/8);
	public static final CudaType INT = new CudaType("int", "int", Integer.SIZE/8);
	public static final CudaType FLOAT = new CudaType("float", "float", Float.SIZE/8);
	public static final CudaType LONG = new CudaType("long", "long", Long.SIZE/8);
	public static final CudaType DOUBLE = new CudaType("double", "double", Double.SIZE/8);
	
	
	// Char vector types
	public static final CudaType CHAR1 = new CudaType("char", "char1", Byte.SIZE/8 * 1);
	public static final CudaType CHAR2 = new CudaType("char", "char2", Byte.SIZE/8 * 2);
	public static final CudaType CHAR3 = new CudaType("char", "char3", Byte.SIZE/8 * 3);
	public static final CudaType CHAR4 = new CudaType("char", "char4", Byte.SIZE/8 * 4);
	public static final CudaType UCHAR1 = new CudaType("char", "uchar1", Byte.SIZE/8 * 1);
	public static final CudaType UCHAR2 = new CudaType("char", "uchar2", Byte.SIZE/8 * 2);
	public static final CudaType UCHAR3 = new CudaType("char", "uchar3", Byte.SIZE/8 * 3);
	public static final CudaType UCHAR4 = new CudaType("char", "uchar4", Byte.SIZE/8 * 4);
	
	// Short vector types
	public static final CudaType SHORT1 = new CudaType("short", "short1", Short.SIZE/8 * 1);
	public static final CudaType SHORT2 = new CudaType("short", "short2", Short.SIZE/8 * 2);
	public static final CudaType SHORT3 = new CudaType("short", "short3", Short.SIZE/8 * 3);
	public static final CudaType SHORT4 = new CudaType("short", "short4", Short.SIZE/8 * 4);
	public static final CudaType USHORT1 = new CudaType("short", "ushort1", Short.SIZE/8 * 1);
	public static final CudaType USHORT2 = new CudaType("short", "ushort2", Short.SIZE/8 * 2);
	public static final CudaType USHORT3 = new CudaType("short", "ushort3", Short.SIZE/8 * 3);
	public static final CudaType USHORT4 = new CudaType("short", "ushort4", Short.SIZE/8 * 4);
	
	// Int vector types
	public static final CudaType INT1 = new CudaType("int", "int1", Integer.SIZE/8 * 1);
	public static final CudaType INT2 = new CudaType("int", "int2", Integer.SIZE/8 * 2);
	public static final CudaType INT3 = new CudaType("int", "int3", Integer.SIZE/8 * 3);
	public static final CudaType INT4 = new CudaType("int", "int4", Integer.SIZE/8 * 4);
	public static final CudaType UINT1 = new CudaType("int", "uint1", Integer.SIZE/8 * 1);
	public static final CudaType UINT2 = new CudaType("int", "uint2", Integer.SIZE/8 * 2);
	public static final CudaType UINT3 = new CudaType("int", "uint3", Integer.SIZE/8 * 3);
	public static final CudaType UINT4 = new CudaType("int", "uint4", Integer.SIZE/8 * 4);
	
	// Long vector types
	public static final CudaType LONG1 = new CudaType("long", "long1", Long.SIZE/8 * 1);
	public static final CudaType LONG2 = new CudaType("long", "long2", Long.SIZE/8 * 2);
	public static final CudaType LONG3 = new CudaType("long", "long3", Long.SIZE/8 * 3);
	public static final CudaType LONG4 = new CudaType("long", "long4", Long.SIZE/8 * 4);
	public static final CudaType ULONG1 = new CudaType("long", "ulong1", Long.SIZE/8 * 1);
	public static final CudaType ULONG2 = new CudaType("long", "ulong2", Long.SIZE/8 * 2);
	public static final CudaType ULONG3 = new CudaType("long", "ulong3", Long.SIZE/8 * 3);
	public static final CudaType ULONG4 = new CudaType("long", "ulong4", Long.SIZE/8 * 4);
	
	// Float vector types
	public static final CudaType FLOAT1 = new CudaType("float", "float1", Float.SIZE/8 * 1);
	public static final CudaType FLOAT2 = new CudaType("float", "float2", Float.SIZE/8 * 2);
	public static final CudaType FLOAT3 = new CudaType("float", "float3", Float.SIZE/8 * 3);
	public static final CudaType FLOAT4 = new CudaType("float", "float4", Float.SIZE/8 * 4);
	
	// Double vector types
	public static final CudaType DOUBLE1 = new CudaType("double", "double1", Double.SIZE/8 * 1);
	public static final CudaType DOUBLE2 = new CudaType("double", "double2", Double.SIZE/8 * 2);
	
	//TODO longlong types and other available types
}
