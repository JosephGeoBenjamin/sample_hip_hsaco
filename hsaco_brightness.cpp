
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

#include <vector>


#define HIP_ASSERT(expression)                                                  \
    {                                                                           \
        hipError_t error = (expression);                                        \
        if (error != hipSuccess) {                                              \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",                    \
                    hipGetErrorString(error), error, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }


template<typename T> // pointer type
void print3d_pln(T matrix, size_t height, size_t width, size_t channel = 1)
{
    std::cout << std::endl;
    for(size_t k=0; k<channel; k++ ){
        std::cout << "[" << std::endl;
        for(size_t i=0; i<height; i++ ){
            for(size_t j=0; j<width; j++ ){
                std::cout << (unsigned int)( matrix[ j + i*width + k*width*height ] ) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
}



//HSACO - Loading call
hipError_t
hipoc_brightness_contrast( void* srcPtr,
                                void* dstPtr,
                                float alpha, int beta,
                                unsigned int height,
                                unsigned int width,
                                unsigned int channel,
                                hipStream_t stream )
{

    // void* argBuffer[7];
    // argBuffer[0] = &srcPtr;
    // argBuffer[1] = &dstPtr;
    // argBuffer[2] = &alpha;
    // argBuffer[3] = &beta;
    // argBuffer[4] = &height;
    // argBuffer[5] = &width;
    // argBuffer[6] = &channel;

    std::vector<void*> argBuffer;
    argBuffer.push_back(&srcPtr);
    argBuffer.push_back(&dstPtr);
    argBuffer.push_back(&alpha);
    argBuffer.push_back(&beta);
    argBuffer.push_back(&height);
    argBuffer.push_back(&width);
    argBuffer.push_back(&channel);

    size_t argSize = 7*sizeof(void*);

    // Note if not working try similar one to cl args setting

    hipModule_t module;
    HIP_ASSERT(hipModuleLoad(&module, "/home/neel/jgeob/rpp-workspace/standalone-arena/toReza/brightness_contrast.cl.o"));
    hipFunction_t function;
    HIP_ASSERT(hipModuleGetFunction(&function, module, "brightness_contrast"));

    hipEvent_t  start = nullptr;
    hipEvent_t stop  = nullptr;

    void* config[] = {  HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                        HIP_LAUNCH_PARAM_END };

    size_t gDim3[3];
    gDim3[0] = width;
    gDim3[1] = height;
    gDim3[2] = channel;
    size_t lDim3[3];
    lDim3[0] = 32;
    lDim3[1] = 32;
    lDim3[2] = channel;

    HIP_ASSERT(hipHccModuleLaunchKernel(brightness_contrast, gDim3[0],
                                       gDim3[1],
                                       gDim3[2],
                                       lDim3[0],
                                       lDim3[1],
                                       lDim3[2],
                                       0, 0,
                                       nullptr,
                                       reinterpret_cast<void**>(&config),
                                       start,
                                       stop));

    HIP_ASSERT(hipDeviceSynchronize());

    return hipSuccess;
}

int main ()
{
    typedef unsigned char TYPE_t;
    TYPE_t* h_a;
    TYPE_t* h_c;
    int height= 20;
    int width = 20;
    int channel = 1;

    size_t n = height * width * channel;
    size_t bytes = n*sizeof(TYPE_t);
    h_a = (TYPE_t*)malloc(bytes);
    h_c = (TYPE_t*)malloc(bytes);
    for (int i; i < n; i++){
        h_a[i] = i;
        h_c[i] = 0;

    }

    print3d_pln(h_a, height, width, channel);
    std::cout << "width:" << width << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "channel:" << channel << std::endl;
    std::cout << "bytes:" << bytes << std::endl;

    TYPE_t* d_a;
    HIP_ASSERT(hipMalloc(&d_a, bytes)) ;
    HIP_ASSERT(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice)) ;
    TYPE_t* d_c;
    HIP_ASSERT(hipMalloc(&d_c, bytes)) ;
    hipStream_t stream;
    HIP_ASSERT(hipStreamCreate(&stream));

    float alpha=1;
    int beta=50;
    //HSACO - Conversion call
    hipoc_brightness_contrast(  d_a, d_c,
                                alpha, beta,
                                height, width, channel,
                                stream );

//----

    HIP_ASSERT(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
    print3d_pln(h_c, height, width, channel);

    free(h_a);
    free(h_c);
    hipFree(d_a);
    hipFree(d_c);

return 0;

}