MLU_MACROS = """
#include <bang.h>
"""

HIP_MACROS = """
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cmath>
#include <hip/hip_fp16.h>
"""


CPP_MACROS = """
#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
typedef unsigned short half;
"""


CUDA_MACROS = """
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdlib.h>

using namespace nvcuda;
"""
