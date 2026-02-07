/* cpu.h  : Author           : Alexander J. Yee */
#ifndef _cpu_H
#define _cpu_H
#include <cstdint>
#include <cstring>
#include <string>

// MODIFIED: Generalized platform detection
#if defined(__ANDROID__)
    // Android specific (existing logic preserved if needed, or treated as generic)
#else
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        #if _WIN32
            #include <Windows.h>
            #include <intrin.h>
        #elif defined(__GNUC__) || defined(__clang__)
            #include <cpuid.h>
            #define _XCR_XFEATURE_ENABLED_MASK  0
        #else
            #error "No cpuid intrinsic defined for compiler."
        #endif
    #else
        // ARM / Generic Platform: No specific intrinsics needed
    #endif
#endif 

namespace fbow{
struct cpu{
    bool Vendor_AMD,Vendor_Intel;//  Vendor
    bool OS_x64,OS_AVX,OS_AVX512;//  OS Features
    bool HW_MMX,HW_x64,HW_ABM,HW_RDRAND,HW_BMI1,HW_BMI2,HW_ADX,HW_PREFETCHWT1,HW_MPX;//  Misc.
    bool HW_SSE,HW_SSE2,HW_SSE3,HW_SSSE3,HW_SSE41,HW_SSE42,HW_SSE4a,HW_AES,HW_SHA;//  SIMD: 128-bit
    bool HW_AVX,HW_XOP,HW_FMA3,HW_FMA4,HW_AVX2;//  SIMD: 256-bit
    bool HW_AVX512_F,HW_AVX512_PF,HW_AVX512_ER,HW_AVX512_CD,HW_AVX512_VL,HW_AVX512_BW,HW_AVX512_DQ,HW_AVX512_IFMA,HW_AVX512_VBMI;//  SIMD: 512-bit
public:
    inline cpu(){  memset(this, 0, sizeof(*this)); }
    inline void detect_host();
    inline bool isSafeAVX(){return  HW_AVX && OS_AVX;}
    inline bool isSafeSSE(){return  HW_SSE ;}
    inline bool isSafeMMX(){return  HW_MMX ;}
    inline void disableAVX(){HW_AVX =false;}
    inline void disableMMX(){HW_MMX =false;}
    inline void disableSSE(){HW_SSE=false;}

    static inline  void cpuid(int32_t out[4], int32_t x);
    static inline std::string get_vendor_string();

private:
    static bool inline detect_OS_x64();
    static bool inline detect_OS_AVX();
    static bool inline detect_OS_AVX512();
    static inline uint64_t xgetbv(unsigned int x);
};

// MODIFIED: Implementation section
#ifdef __ANDROID__
    void cpu::cpuid(int32_t out[4], int32_t x){}
    // Android helpers... (rest of Android logic handled implicitly or extended below)
#endif

// x86 / x64 Implementations
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #if _WIN32
        void  cpu::cpuid(int32_t out[4], int32_t x){ __cpuidex(out, x, 0); }
        uint64_t  cpu::xgetbv(unsigned int x){ return _xgetbv(x); }
        typedef BOOL (WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
        static inline BOOL IsWow64() {
            BOOL bIsWow64 = FALSE;
            LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS) GetProcAddress( GetModuleHandle(TEXT("kernel32")), "IsWow64Process");
            if (NULL == fnIsWow64Process) return FALSE;
            if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64)) return FALSE;
            return bIsWow64 ;
        }
        bool  cpu::detect_OS_x64(){
        #ifdef _M_X64
            return true;
        #else
            return IsWow64() != 0;
        #endif
        }
    #elif defined(__GNUC__) || defined(__clang__)
        void cpu::cpuid(int32_t out[4], int32_t x){ __cpuid_count(x, 0, out[0], out[1], out[2], out[3]); }
        uint64_t cpu::xgetbv(unsigned int index){ uint32_t eax, edx;  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index)); return ((uint64_t)edx << 32) | eax;}
        bool  cpu::detect_OS_x64(){ return true;}
    #endif

    bool cpu::detect_OS_AVX(){
        //  Copied from: http://stackoverflow.com/a/22521619/922184
        bool avxSupported = false;
        int cpuInfo[4]; cpuid(cpuInfo, 1);
        bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
        bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;
        if (osUsesXSAVE_XRSTORE && cpuAVXSuport) { uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK); avxSupported = (xcrFeatureMask & 0x6) == 0x6;}
        return avxSupported;
    }
    bool cpu::detect_OS_AVX512(){
        if (!detect_OS_AVX()) return false;
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return (xcrFeatureMask & 0xe6) == 0xe6;
    }
#else 
// MODIFIED: ARM / Generic Implementation
// Provide dummy implementations that report no features
    #ifndef __ANDROID__ // Android is handled above, avoiding redefinition if mixed
    void cpu::cpuid(int32_t out[4], int32_t x){ memset(out, 0, 4*sizeof(int32_t)); }
    #endif
    
    // Generic stubs
    #if !defined(_WIN32) && !defined(__GNUC__) && !defined(__clang__) && !defined(__ANDROID__)
     // Ensure we don't break if compiler is weird, but usually one of above is true.
     // If we are here, we are likely ARM Linux/Mac.
    #endif

    // Stub out internal helpers
    #ifndef __ANDROID__
    uint64_t cpu::xgetbv(unsigned int x){ return 0; }
    bool cpu::detect_OS_x64(){ return sizeof(void*) == 8; }
    bool cpu::detect_OS_AVX(){ return false; }
    bool cpu::detect_OS_AVX512(){ return false; }
    #endif
#endif

// Shared / Generic methods
std::string cpu::get_vendor_string(){ 
    int32_t CPUInfo[4]; char name[13];
    cpuid(CPUInfo, 0); 
    memcpy(name + 0, &CPUInfo[1], 4);
    memcpy(name + 4, &CPUInfo[3], 4); 
    memcpy(name + 8, &CPUInfo[2], 4); 
    name[12] = '\0'; 
    return name;
}

void cpu::detect_host(){
    memset(this, 0, sizeof(*this)); // Reset all flags to false by default

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // Only run x86 detection logic on x86 platforms
    #ifndef __ANDROID__
        OS_x64 = detect_OS_x64();
        OS_AVX = detect_OS_AVX();
        OS_AVX512 = detect_OS_AVX512();
        //  Vendor
        std::string vendor(get_vendor_string());
        if (vendor == "GenuineIntel"){ Vendor_Intel = true;}
        else if (vendor == "AuthenticAMD"){ Vendor_AMD = true; }
        int info[4];
        cpuid(info, 0);
        int nIds = info[0];
        cpuid(info, 0x80000000);
        uint32_t nExIds = info[0];
        //  Detect Features
        if (nIds >= 0x00000001){ cpuid(info, 0x00000001); HW_MMX    = (info[3] & ((int)1 << 23)) != 0; HW_SSE    = (info[3] & ((int)1 << 25)) != 0; HW_SSE2   = (info[3] & ((int)1 << 26)) != 0; HW_SSE3   = (info[2] & ((int)1 <<  0)) != 0; HW_SSSE3  = (info[2] & ((int)1 <<  9)) != 0; HW_SSE41  = (info[2] & ((int)1 << 19)) != 0; HW_SSE42  = (info[2] & ((int)1 << 20)) != 0; HW_AES    = (info[2] & ((int)1 << 25)) != 0; HW_AVX    = (info[2] & ((int)1 << 28)) != 0; HW_FMA3   = (info[2] & ((int)1 << 12)) != 0; HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;    }
        if (nIds >= 0x00000007){ cpuid(info, 0x00000007); HW_AVX2         = (info[1] & ((int)1 <<  5)) != 0; HW_BMI1         = (info[1] & ((int)1 <<  3)) != 0; HW_BMI2         = (info[1] & ((int)1 <<  8)) != 0; HW_ADX          = (info[1] & ((int)1 << 19)) != 0; HW_MPX          = (info[1] & ((int)1 << 14)) != 0; HW_SHA          = (info[1] & ((int)1 << 29)) != 0; HW_PREFETCHWT1  = (info[2] & ((int)1 <<  0)) != 0; HW_AVX512_F     = (info[1] & ((int)1 << 16)) != 0; HW_AVX512_CD    = (info[1] & ((int)1 << 28)) != 0; HW_AVX512_PF    = (info[1] & ((int)1 << 26)) != 0; HW_AVX512_ER    = (info[1] & ((int)1 << 27)) != 0; HW_AVX512_VL    = (info[1] & ((int)1 << 31)) != 0; HW_AVX512_BW    = (info[1] & ((int)1 << 30)) != 0; HW_AVX512_DQ    = (info[1] & ((int)1 << 17)) != 0; HW_AVX512_IFMA  = (info[1] & ((int)1 << 21)) != 0; HW_AVX512_VBMI  = (info[2] & ((int)1 <<  1)) != 0;    }
        if (nExIds >= 0x80000001){ cpuid(info, 0x80000001); HW_x64   = (info[3] & ((int)1 << 29)) != 0; HW_ABM   = (info[2] & ((int)1 <<  5)) != 0; HW_SSE4a = (info[2] & ((int)1 <<  6)) != 0; HW_FMA4  = (info[2] & ((int)1 << 16)) != 0; HW_XOP   = (info[2] & ((int)1 << 11)) != 0; }
    #endif
#else
    // ARM / Generic: Leave all feature flags as false.
    // This forces fbow.cpp to use the generic fallbacks (L1_x32, L2_generic)
    // which use standard C++ and no x86 intrinsics.
#endif
}

}
#endif